import torch
import torch.nn as nn
from typing import Dict
from .base_model import BaseHangmanModel
import pytorch_lightning as pl
from torch.optim.lr_scheduler import OneCycleLR

class CNNLSTMModel(BaseHangmanModel, pl.LightningModule):
    def __init__(
        self,
        max_word_length: int = 30,
        cnn_channels: list = [28, 32, 64],
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        total_steps: int = 1000
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Model parameters
        self.max_word_length = max_word_length
        self.learning_rate = learning_rate
        self.total_steps = total_steps
        
        # CNN layers with residual connections
        cnn_layers = []
        for i in range(len(cnn_channels) - 1):
            cnn_layers.extend([
                nn.Conv1d(cnn_channels[i], cnn_channels[i+1], kernel_size=3, padding=1),
                nn.BatchNorm1d(cnn_channels[i+1]),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            # Add residual connection if channels match
            if cnn_channels[i] == cnn_channels[i+1]:
                cnn_layers.append(ResidualConnection())
                
        self.cnn = nn.Sequential(*cnn_layers)
        
        # Bidirectional LSTM with skip connections
        self.lstm = LSTMWithSkipConnections(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = MultiHeadAttention(
            lstm_hidden * 2,  # * 2 for bidirectional
            num_heads=8
        )
        
        # Output layers with skip connections
        self.output_layers = nn.Sequential(
            nn.Linear(lstm_hidden * 2 + 26, lstm_hidden),
            nn.LayerNorm(lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            ResidualLinear(lstm_hidden, lstm_hidden),
            nn.LayerNorm(lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, 26)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Process word state through CNN
        word_state = x['word_state'].transpose(1, 2)  # [batch, 28, length]
        conv_out = self.cnn(word_state)
        conv_out = conv_out.transpose(1, 2)  # [batch, length, channels]
        
        # Pack sequence for LSTM
        lengths = x['word_length'].squeeze(-1).cpu()  # Remove extra dimension and move to CPU
        # Sort sequences by length for packing
        lengths, sort_idx = lengths.sort(descending=True)
        conv_out = conv_out[sort_idx]
        
        # Pack the sorted sequences
        packed = nn.utils.rnn.pack_padded_sequence(
            conv_out, lengths.clamp(max=conv_out.size(1)), 
            batch_first=True, enforce_sorted=True  # Now sequences are sorted
        )
        
        # LSTM processing
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        
        # Restore original batch order
        _, unsort_idx = sort_idx.sort()
        lstm_out = lstm_out[unsort_idx]
        
        # Apply attention
        attended = self.attention(lstm_out)
        
        # Global max pooling and average pooling
        max_pooled = torch.max(attended, dim=1)[0]  # [batch, lstm_hidden*2]
        avg_pooled = torch.mean(attended, dim=1)    # [batch, lstm_hidden*2]
        
        # Combine features
        combined = torch.cat([
            max_pooled,                # [batch, lstm_hidden*2]
            x['alphabet_state']        # [batch, 26]
        ], dim=1)                     # Result: [batch, lstm_hidden*2 + 26]
        
        return self.output_layers(combined)
    
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
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            total_steps=self.total_steps,
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