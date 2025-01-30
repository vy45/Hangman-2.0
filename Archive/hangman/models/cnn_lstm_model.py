import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from .base_model import BaseHangmanModel, ModelConfig
from dataclasses import dataclass
from torch.optim.lr_scheduler import OneCycleLR

@dataclass
class CNNLSTMConfig(ModelConfig):
    """CNN-LSTM specific configuration"""
    cnn_channels: int = 256
    cnn_kernel_size: int = 3
    lstm_layers: int = 2
    bidirectional: bool = True
    
    def __post_init__(self):
        """Set derived parameters"""
        self.lstm_hidden = self.hidden_dim
        self.output_dim = self.lstm_hidden * 2 if self.bidirectional else self.lstm_hidden

class CNNLSTMModel(BaseHangmanModel):
    """CNN-LSTM model for Hangman with packed sequence handling"""
    
    def __init__(self, config: CNNLSTMConfig):
        """Initialize CNN-LSTM architecture"""
        super().__init__(config)
        
        # Initialize CNN
        self.cnn = nn.Sequential(
            nn.Conv1d(
                config.input_dim,
                config.cnn_channels,
                kernel_size=config.cnn_kernel_size,
                padding=config.cnn_kernel_size//2
            ),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # Initialize LSTM
        self.lstm = nn.LSTM(
            input_size=config.cnn_channels,
            hidden_size=config.lstm_hidden,
            num_layers=config.lstm_layers,
            dropout=config.dropout if config.lstm_layers > 1 else 0,
            batch_first=True,
            bidirectional=config.bidirectional
        )
        
        # Initialize output layer
        self.output = nn.Linear(config.output_dim + 26, 26)  # Added +26 for alphabet_state
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize network weights"""
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass with packed sequence handling"""
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
        elif L > self.config.max_length:
            word_state = word_state[:, :self.config.max_length]
            attention_mask = attention_mask[:, :self.config.max_length]
        
        # CNN expects [B, C, L] format
        x = word_state.transpose(1, 2)  # [B, input_dim, L]
        x = self.cnn(x)  # [B, hidden_dim, L]
        x = x.transpose(1, 2)  # [B, L, hidden_dim]
        
        # Get lengths from attention mask
        lengths = attention_mask.sum(dim=1).cpu()
        
        # Pack sequence for LSTM
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        
        # LSTM processing
        lstm_out, _ = self.lstm(packed)
        
        # Unpack and handle padding
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=True, total_length=self.config.max_length
        )
        
        # Pool using attention mask
        mask = attention_mask.unsqueeze(-1).float()  # [B, L, 1]
        pooled = (unpacked * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # [B, hidden_dim*2]
        
        # Combine with alphabet state
        combined = torch.cat([pooled, alphabet_state], dim=1)  # [B, hidden_dim*2 + 26]
        
        return self.output(combined)  # [B, 26]
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step with metrics logging"""
        logits = self(batch)
        loss = F.cross_entropy(logits, batch['target_distribution'])
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        accuracy = (logits.argmax(dim=1) == batch['target_distribution'].argmax(dim=1)).float().mean()
        self.log('train_accuracy', accuracy, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step with metrics logging"""
        logits = self(batch)
        loss = F.cross_entropy(logits, batch['target_distribution'])
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        accuracy = (logits.argmax(dim=1) == batch['target_distribution'].argmax(dim=1)).float().mean()
        self.log('val_accuracy', accuracy, prog_bar=True)
        
        return {
            'val_loss': loss,
            'val_accuracy': accuracy
        }
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.config.learning_rate,
            total_steps=self.hparams.total_steps,
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }

# Helper modules
class ResidualConnection(nn.Module):
    def forward(self, x):
        return x + x

class ResidualLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        return x + self.activation(self.norm(self.linear(x)))

class LSTMWithSkipConnections(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, bidirectional):
        super().__init__()
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                input_size if i == 0 else hidden_size * 2,
                hidden_size,
                1,
                batch_first=True,
                bidirectional=bidirectional
            )
            for i in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, packed_input):
        # Unpack the input
        unpacked, lengths = nn.utils.rnn.pad_packed_sequence(packed_input, batch_first=True)
        output = unpacked
        
        for lstm in self.lstm_layers:
            new_output, _ = lstm(output)
            if output.size() == new_output.size():
                output = output + new_output
            else:
                output = new_output
            output = self.dropout(output)
            
        # Repack the output
        output = nn.utils.rnn.pack_padded_sequence(
            output, lengths, batch_first=True, enforce_sorted=False
        )
        return output, None

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True
        )
        
    def forward(self, x):
        attended, _ = self.attention(x, x, x)
        return attended 