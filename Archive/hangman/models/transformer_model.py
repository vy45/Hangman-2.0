import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from .base_model import BaseHangmanModel, ModelConfig
import pytorch_lightning as pl
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataclasses import dataclass

@dataclass
class TransformerConfig(ModelConfig):
    """Transformer-specific configuration"""
    num_heads: int = 8
    num_layers: int = 4
    ff_dim: Optional[int] = None  # Will default to 4 * hidden_dim
    
    def __post_init__(self):
        """Set derived parameters"""
        self.ff_dim = self.ff_dim or 4 * self.hidden_dim

class TransformerModel(BaseHangmanModel):
    """Transformer model for Hangman"""
    
    def __init__(self, config: TransformerConfig):
        """Initialize transformer architecture
        
        Args:
            config: Transformer configuration
        """
        super().__init__(config)
        
        # Initialize embeddings
        self.embedding = nn.Linear(config.input_dim, config.hidden_dim)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, config.max_length, config.hidden_dim)
        )
        
        # Initialize transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=config.num_layers
        )
        
        # Initialize output layers
        self.combine_layer = nn.Sequential(
            nn.Linear(config.hidden_dim + 26, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 26)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass of transformer
        
        Args:
            batch: Input batch containing:
                - word_state: [B, L, input_dim]
                - attention_mask: [B, L]
                - alphabet_state: [B, 26]
                
        Returns:
            Logits tensor [B, 26]
        """
        # Get batch components
        word_state = batch['word_state']  # [B, L, input_dim]
        attention_mask = batch['attention_mask']  # [B, L]
        alphabet_state = batch['alphabet_state']  # [B, 26]
        
        # Handle padding to max_length
        B, L, D = word_state.shape
        if L < self.config.max_length:
            padding = torch.zeros(B, self.config.max_length - L, D, device=word_state.device)
            word_state = torch.cat([word_state, padding], dim=1)
            mask_padding = torch.zeros(B, self.config.max_length - L, dtype=bool, device=attention_mask.device)
            attention_mask = torch.cat([attention_mask, mask_padding], dim=1)
        
        # Embedding with positional encoding
        x = self.embedding(word_state)  # [B, L, hidden_dim]
        x = x + self.pos_embedding[:, :x.size(1)]  # Add positional encoding
        
        # Create attention mask for transformer
        padding_mask = ~attention_mask  # Invert for transformer
        
        # Transformer layers
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        
        # Pool valid tokens only
        mask = attention_mask.unsqueeze(-1).float()  # [B, L, 1]
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # [B, hidden_dim]
        
        # Combine with alphabet state
        combined = torch.cat([pooled, alphabet_state], dim=1)  # [B, hidden_dim + 26]
        
        return self.combine_layer(combined)  # [B, 26]
    
    def training_step(self, batch, batch_idx):
        logits = self(batch)
        loss = nn.CrossEntropyLoss()(logits, batch['target_distribution'])
        
        # Log metrics
        self.log('train_loss', loss)
        accuracy = (logits.argmax(dim=1) == batch['target_distribution'].argmax(dim=1)).float().mean()
        self.log('train_accuracy', accuracy)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        loss = nn.CrossEntropyLoss()(logits, batch['target_distribution'])
        
        # Log metrics
        self.log('val_loss', loss)
        accuracy = (logits.argmax(dim=1) == batch['target_distribution'].argmax(dim=1)).float().mean()
        self.log('val_accuracy', accuracy)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.warmup_steps,
            eta_min=self.learning_rate / 10
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        } 