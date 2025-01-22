import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from .base_model import BaseHangmanModel
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau

class MLPAttentionModel(BaseHangmanModel, pl.LightningModule):
    def __init__(
        self,
        max_word_length: int = 30,
        hidden_dims: list = [64, 128, 64],
        num_heads: int = 4,
        dropout: float = 0.1,
        learning_rate: float = 1e-3
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Initial word processing
        self.word_encoder = nn.Sequential(
            nn.Linear(28, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Position encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, max_word_length, hidden_dims[0]))
        
        # Multi-head self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dims[0],
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attention_norm = nn.LayerNorm(hidden_dims[0])
        
        # Deep MLP layers with residual connections
        self.mlp_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.mlp_layers.append(
                ResidualMLPBlock(
                    hidden_dims[i],
                    hidden_dims[i + 1],
                    dropout=dropout
                )
            )
        
        # Final processing - corrected input dimension
        final_dim = hidden_dims[-1] + 26  # Only one pooled output + alphabet state
        self.final_layers = nn.Sequential(
            nn.Linear(final_dim, hidden_dims[-1] // 2),
            nn.LayerNorm(hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1] // 2, hidden_dims[-1] // 4),
            nn.LayerNorm(hidden_dims[-1] // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1] // 4, 26)
        )
        
        self.learning_rate = learning_rate
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Process word state
        word_state = x['word_state']  # [batch, length, 28]
        
        # Initial encoding
        encoded = self.word_encoder(word_state)
        
        # Add positional encoding
        encoded = encoded + self.pos_embedding[:, :encoded.size(1)]
        
        # Create attention mask for padding
        padding_mask = (word_state[:, :, -1] == 1)  # True for padding positions
        
        # Self-attention with residual connection
        attended, _ = self.self_attention(
            encoded, encoded, encoded,
            key_padding_mask=padding_mask
        )
        encoded = self.attention_norm(encoded + attended)
        
        # Process through MLP layers
        features = encoded
        for mlp_layer in self.mlp_layers:
            features = mlp_layer(features)
            
        # Global average pooling only (removed max pooling to fix dimensions)
        mask = ~padding_mask.unsqueeze(-1)
        avg_pooled = (features * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        
        # Combine with alphabet state
        combined = torch.cat([
            avg_pooled,           # [batch, hidden_dims[-1]]
            x['alphabet_state']   # [batch, 26]
        ], dim=1)
        
        return self.final_layers(combined)
    
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