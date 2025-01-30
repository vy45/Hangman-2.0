from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from dataclasses import dataclass
from .base_model import BaseHangmanModel, ModelConfig

@dataclass
class GNNConfig(ModelConfig):
    """GNN-specific configuration"""
    num_layers: int = 3
    neighbor_distance: int = 2  # Connect nodes within this distance
    use_edge_features: bool = True
    pooling_type: str = 'mean'  # 'mean', 'max', or 'attention'
    
    def __post_init__(self):
        """Validate configuration"""
        valid_pooling = {'mean', 'max', 'attention'}
        if self.pooling_type not in valid_pooling:
            raise ValueError(f"pooling_type must be one of {valid_pooling}")

class GNNModel(BaseHangmanModel):
    """Graph Neural Network for Hangman"""
    
    def __init__(self, config: GNNConfig):
        """Initialize GNN architecture"""
        super().__init__(config)
        
        # Initialize embeddings
        self.embedding = nn.Linear(config.input_dim, config.hidden_dim)
        
        # Initialize GNN layers
        self.gnn_layers = nn.ModuleList([
            GCNConv(config.hidden_dim, config.hidden_dim)
            for _ in range(config.num_layers)
        ])
        
        # Initialize output layers
        self.output = nn.Sequential(
            nn.Linear(config.hidden_dim + 26, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 26)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _create_graph_batch(
        self,
        word_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create batched graph from word states"""
        batch_size, seq_len, _ = word_states.shape
        device = word_states.device
        
        # Handle padding to max_length first
        if seq_len > self.config.max_length:
            word_states = word_states[:, :self.config.max_length]
            attention_mask = attention_mask[:, :self.config.max_length]
            seq_len = self.config.max_length
        
        # Create edges within neighbor_distance
        edges = []
        batch_idx = []
        node_offset = 0  # Track total valid nodes processed
        
        for b in range(batch_size):
            # Get valid node indices for this sequence
            valid = attention_mask[b].nonzero().squeeze(-1)
            valid_len = len(valid)
            
            # Add edges between valid nodes using local offsets
            for i in range(valid_len):
                for j in range(max(0, i - self.config.neighbor_distance),
                             min(valid_len, i + self.config.neighbor_distance + 1)):
                    if i != j:
                        edges.append([
                            i + node_offset,
                            j + node_offset
                        ])
            
            # Track batch membership
            batch_idx.extend([b] * valid_len)
            node_offset += valid_len
        
        # Convert to tensors
        if not edges:
            edge_index = torch.zeros((2, 0), device=device, dtype=torch.long)
        else:
            edge_index = torch.tensor(edges, device=device).t().contiguous()
        batch = torch.tensor(batch_idx, device=device)
        
        # Get valid node features
        valid_mask = attention_mask.bool()
        node_features = word_states[valid_mask]
        
        return node_features, edge_index, batch
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass of GNN"""
        # Get batch components
        word_state = batch['word_state']  # [B, L, input_dim]
        attention_mask = batch['attention_mask']  # [B, L]
        alphabet_state = batch['alphabet_state']  # [B, 26]
        
        # Create graph batch
        node_features, edge_index, batch_idx = self._create_graph_batch(
            word_state, attention_mask
        )
        
        # Initial node embedding
        x = self.embedding(node_features)  # [N, hidden_dim]
        
        # Apply GNN layers
        for conv in self.gnn_layers:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.config.dropout, training=self.training)
        
        # Global pooling
        pooled = global_mean_pool(x, batch_idx)  # [B, hidden_dim]
        
        # Combine with alphabet state
        combined = torch.cat([pooled, alphabet_state], dim=1)  # [B, hidden_dim + 26]
        
        return self.output(combined)  # [B, 26]
    
    def training_step(self, batch, batch_idx):
        logits = self(batch)
        loss = F.cross_entropy(logits, batch['target_distribution'])
        
        self.log('train_loss', loss)
        accuracy = (logits.argmax(dim=1) == batch['target_distribution'].argmax(dim=1)).float().mean()
        self.log('train_accuracy', accuracy)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        loss = F.cross_entropy(logits, batch['target_distribution'])
        
        # Log metrics exactly like training_step
        self.log('val_loss', loss, prog_bar=True)
        accuracy = (logits.argmax(dim=1) == batch['target_distribution'].argmax(dim=1)).float().mean()
        self.log('val_accuracy', accuracy, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
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