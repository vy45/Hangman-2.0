import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from .base_model import BaseHangmanModel, ModelConfig
from dataclasses import dataclass
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau

@dataclass
class MLPAttentionConfig(ModelConfig):
    """MLP-Attention specific configuration"""
    num_layers: int = 3
    attention_heads: int = 4
    attention_dropout: float = 0.1
    use_layer_norm: bool = True
    activation: str = 'relu'
    
    def __post_init__(self):
        """Validate configuration"""
        valid_activations = {'relu', 'gelu', 'silu'}
        if self.activation not in valid_activations:
            raise ValueError(f"activation must be one of {valid_activations}")

class MLPAttentionModel(BaseHangmanModel):
    """MLP with self-attention for Hangman"""
    
    def __init__(self, config: MLPAttentionConfig):
        """Initialize MLP-Attention architecture"""
        super().__init__(config)
        
        # Initialize embeddings
        self.embedding = nn.Linear(config.input_dim, config.hidden_dim)
        
        # Initialize attention
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.attention_heads,
            dropout=config.attention_dropout,
            batch_first=True
        )
        
        # Initialize MLP layers
        mlp_layers = []
        for _ in range(config.num_layers):
            mlp_layers.extend([
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim) if config.use_layer_norm else nn.Identity(),
                self._get_activation(config.activation),
                nn.Dropout(config.dropout)
            ])
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Initialize output layer
        self.output = nn.Linear(config.hidden_dim + 26, 26)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name"""
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU()
        }
        return activations[name]
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass with attention and MLP"""
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
        
        # Initial embedding
        x = self.embedding(word_state)  # [B, L, hidden_dim]
        
        # Self-attention
        attention_mask = ~attention_mask  # Invert for attention
        attended, _ = self.attention(
            x, x, x,
            key_padding_mask=attention_mask
        )
        
        # Residual connection
        x = x + attended
        
        # MLP processing
        x = self.mlp(x)  # [B, L, hidden_dim]
        
        # Global average pooling
        mask = (~attention_mask).float().unsqueeze(-1)  # [B, L, 1]
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # [B, hidden_dim]
        
        # Combine with alphabet state
        combined = torch.cat([pooled, alphabet_state], dim=1)  # [B, hidden_dim + 26]
        
        return self.output(combined)  # [B, 26]
    
    def training_step(self, batch, batch_idx):
        logits = self(batch)
        loss = F.cross_entropy(logits, batch['target_distribution'])
        
        # Log metrics
        self.log('train_loss', loss)
        accuracy = (logits.argmax(dim=1) == batch['target_distribution'].argmax(dim=1)).float().mean()
        self.log('train_accuracy', accuracy)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        loss = F.cross_entropy(logits, batch['target_distribution'])
        
        # Log metrics
        self.log('val_loss', loss)
        accuracy = (logits.argmax(dim=1) == batch['target_distribution'].argmax(dim=1)).float().mean()
        self.log('val_accuracy', accuracy)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }

class ResidualMLPBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.norm1 = nn.LayerNorm(out_features)
        self.norm2 = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
        # Optional projection for residual connection if dimensions don't match
        self.projection = None
        if in_features != out_features:
            self.projection = nn.Linear(in_features, out_features)
            
    def forward(self, x):
        # First sub-block
        identity = x
        x = self.linear1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Second sub-block
        x = self.linear2(x)
        x = self.norm2(x)
        
        # Residual connection
        if self.projection is not None:
            identity = self.projection(identity)
        x = x + identity
        
        x = self.activation(x)
        x = self.dropout(x)
        
        return x 