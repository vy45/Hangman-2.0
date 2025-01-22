import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from typing import Dict
from .base_model import BaseHangmanModel
import pytorch_lightning as pl

class GNNModel(BaseHangmanModel, pl.LightningModule):
    def __init__(
        self,
        node_features: int = 28,
        hidden_channels: list = [32, 64],   # Reduced from [64, 128, 256]
        dropout: float = 0.1,
        learning_rate: float = 1e-3
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # GNN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Input layer
        self.convs.append(GCNConv(node_features, hidden_channels[0]))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels[0]))
        
        # Hidden layers
        for i in range(len(hidden_channels) - 1):
            self.convs.append(GCNConv(hidden_channels[i], hidden_channels[i + 1]))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels[i + 1]))
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_channels[-1] * 2 + 26, hidden_channels[-1]),
            nn.LayerNorm(hidden_channels[-1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels[-1], hidden_channels[-1] // 2),
            nn.LayerNorm(hidden_channels[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels[-1] // 2, 26)
        )
        
        self.learning_rate = learning_rate
        
    def _create_graph_data(self, word_state: torch.Tensor) -> tuple:
        batch_size, seq_len, _ = word_state.shape
        device = word_state.device
        
        # Create edges (connect each position to its neighbors)
        edges = []
        batch = []
        for b in range(batch_size):
            for i in range(seq_len):
                for j in range(max(0, i-2), min(seq_len, i+3)):  # Connect to 2 neighbors each side
                    if i != j:
                        edges.append([i + b*seq_len, j + b*seq_len])
            batch.extend([b] * seq_len)
            
        edge_index = torch.tensor(edges, device=device).t().contiguous()
        batch = torch.tensor(batch, device=device)
        
        # Reshape node features
        x = word_state.reshape(-1, word_state.size(-1))
        
        return x, edge_index, batch
        
    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Create graph data
        node_features, edge_index, batch = self._create_graph_data(x['word_state'])
        
        # Apply GNN layers
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            node_features = conv(node_features, edge_index)
            node_features = batch_norm(node_features)
            node_features = F.relu(node_features)
            node_features = F.dropout(node_features, p=self.hparams.dropout, training=self.training)
        
        # Move tensors to CPU for unsupported operations
        node_features_cpu = node_features.cpu()
        batch_cpu = batch.cpu()
        
        # Global pooling on CPU
        max_pooled = global_max_pool(node_features_cpu, batch_cpu)
        mean_pooled = global_mean_pool(node_features_cpu, batch_cpu)
        
        # Move back to original device
        max_pooled = max_pooled.to(node_features.device)
        mean_pooled = mean_pooled.to(node_features.device)
        
        # Combine with alphabet state
        combined = torch.cat([max_pooled, mean_pooled, x['alphabet_state']], dim=1)
        
        return self.output_layers(combined)
    
    def training_step(self, batch, batch_idx):
        logits = self(batch)
        loss = F.cross_entropy(logits, batch['target_distribution'])
        
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
        return optimizer 