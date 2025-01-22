import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from .base_model import BaseHangmanModel
import pytorch_lightning as pl
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR

class TransformerModel(BaseHangmanModel, pl.LightningModule):
    def __init__(
        self,
        max_word_length: int = 30,
        hidden_dim: int = 128,              # Reduced from 384
        num_heads: int = 4,                 # Reduced from 12
        num_layers: int = 2,                # Reduced from 6
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
        warmup_steps: int = 100             # Reduced from 1000
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Ensure hidden_dim is divisible by num_heads
        assert hidden_dim % num_heads == 0, f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        
        # Model parameters
        self.max_word_length = max_word_length
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        
        # Word embedding with larger capacity
        self.word_embedding = nn.Sequential(
            nn.Linear(28, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Learned position embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, max_word_length, hidden_dim))
        
        # Transformer encoder with more layers and heads
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-norm architecture
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim)
        )
        
        # Enhanced combination layers
        self.combine_layer = nn.Sequential(
            nn.Linear(hidden_dim + 26, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 26)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Process word state [batch_size, max_length, 28]
        word_state = x['word_state']
        if word_state.size(1) < self.max_word_length:
            padding = torch.zeros(
                word_state.size(0),
                self.max_word_length - word_state.size(1),
                word_state.size(2),
                device=word_state.device
            )
            word_state = torch.cat([word_state, padding], dim=1)
        
        # Enhanced embedding process
        embedded = self.word_embedding(word_state)
        embedded = embedded + self.pos_embedding[:, :embedded.size(1)]
        
        # Create padding mask
        padding_mask = (word_state[:, :, -1] == 1)
        
        # Transformer processing with residual connection
        transformed = self.transformer(embedded, src_key_padding_mask=padding_mask)
        
        # Sophisticated pooling
        mask = ~padding_mask.unsqueeze(-1)
        pooled = (transformed * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        
        # Combine with alphabet state
        combined = torch.cat([pooled, x['alphabet_state']], dim=1)
        
        return self.combine_layer(combined)
    
    def training_step(self, batch, batch_idx):
        logits = self(batch)
        loss = nn.CrossEntropyLoss()(logits, batch['target_distribution'])
        
        # Log metrics
        self.log('train_loss', loss)
        accuracy = (logits.argmax(dim=1) == batch['target_distribution'].argmax(dim=1)).float().mean()
        self.log('train_accuracy', accuracy)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.warmup_steps,
            eta_min=self.learning_rate / 10
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        } 