import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb
from typing import List, Type, Dict, Optional, Any, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import time
from dataclasses import dataclass

from hangman.models.transformer_model import TransformerModel
from hangman.models.cnn_lstm_model import CNNLSTMModel
from hangman.models.gnn_model import GNNModel
from hangman.models.mlp_attention_model import MLPAttentionModel
from hangman.utils.benchmarking import ModelBenchmark
from hangman.data.word_processor import WordProcessor
from hangman.data.simulator import HangmanSimulator
from hangman.data.dataset import get_dataloaders, HangmanDataset, HangmanValidationDataset

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs"""
    quick_mode: bool = True
    data_percentage: float = 5.0 if quick_mode else 100.0
    max_epochs: int = 1 if quick_mode else 100
    batch_size: int = 32
    patience: int = 2 if quick_mode else 10
    log_every_n_steps: int = 10 if quick_mode else 50
    wandb_logging: bool = True
    max_length: int = 30
    input_dim: int = 28
    hidden_dim: int = 256
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100

    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.quick_mode:
            self.data_percentage = 5.0
            self.max_epochs = 1  # Set to 1 epoch
            self.patience = 2
            self.log_every_n_steps = 10
        
        # Validate learning parameters
        if self.learning_rate <= 0:
            raise ValueError(f"Learning rate must be positive, got {self.learning_rate}")
        if self.weight_decay < 0:
            raise ValueError(f"Weight decay must be non-negative, got {self.weight_decay}")
        if self.warmup_steps <= 0:
            raise ValueError(f"Warmup steps must be positive, got {self.warmup_steps}")

class ModelBenchmarker:
    def __init__(
        self,
        config: BenchmarkConfig,
        model_classes: List[Type],
        data_dir: Path,
    ):
        """Initialize benchmarking environment
        
        Args:
            config: Benchmark configuration
            model_classes: List of model classes to benchmark
            data_dir: Directory containing dataset
        """
        # Initialize configuration
        self.config = config
        self.model_classes = model_classes
        self.data_dir = Path(data_dir)
        
        # Initialize tracking
        self.results: Dict[str, Dict[str, Any]] = {}
        self.start_time: float = time.time()
        
        # Initialize data
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        
        # Initialize wandb
        self.wandb_run: Optional[Any] = None
        
        # Validate setup
        self._validate_environment()
        
    def _validate_environment(self) -> None:
        """Validate all required components are available"""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # Check CUDA availability
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Validate model classes
        for model_class in self.model_classes:
            if not hasattr(model_class, 'forward'):
                raise ValueError(f"Model {model_class.__name__} missing forward method")

    def train_model(
        self,
        model_class: Type,
        model_name: str,
        config: Any
    ) -> Dict[str, Any]:
        """Train and evaluate a single model"""
        metrics = {
            'train_loss': [],
            'completion_rate': [],
            'val_accuracy': [],
            'validation_times': [],
            'epoch_times': [],      # Track per-epoch time
            'batch_times': [],      # Track per-batch time
            'total_time': 0.0       # Total training time
        }
        
        try:
            model_start = time.time()
            model = model_class(config=config).to(self.device)
            
            # Log model size
            num_params = sum(p.numel() for p in model.parameters())
            print(f"\nModel {model_name} has {num_params:,} parameters")
            
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
            criterion = nn.CrossEntropyLoss()
            
            # Training loop
            steps = 0
            for epoch in range(self.config.max_epochs):
                epoch_start = time.time()
                model.train()
                
                for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch}")):
                    batch_start = time.time()
                    try:
                        batch = {k: v.to(self.device) for k, v in batch.items()}
                        batch['target_distribution'] = batch.pop('target')
                        
                        # Forward pass
                        outputs = model(batch)
                        loss = criterion(outputs, batch['target_distribution'])
                        
                        # Backward pass
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                        # Track batch time
                        batch_time = time.time() - batch_start
                        metrics['batch_times'].append(batch_time)
                        
                        # Log metrics every N steps
                        steps += 1
                        if steps % self.config.log_every_n_steps == 0:
                            metrics['train_loss'].append(loss.item())
                            
                            # Time validation
                            val_start = time.time()
                            completion_rate, val_accuracy = self._validate_model(model)
                            val_time = time.time() - val_start
                            
                            metrics['completion_rate'].append(completion_rate)
                            metrics['val_accuracy'].append(val_accuracy)
                            metrics['validation_times'].append(val_time)
                            
                            print(f"\nStep {steps}:")
                            print(f"Batch Time: {batch_time:.3f}s")
                            print(f"Validation Time: {val_time:.2f}s")
                            print(f"Completion Rate: {completion_rate:.3f}")
                            print(f"Validation Accuracy: {val_accuracy:.3f}")
                            
                            if self.config.wandb_logging:
                                wandb.log({
                                    'train_loss': loss.item(),
                                    'completion_rate': completion_rate,
                                    'val_accuracy': val_accuracy,
                                    'validation_time': val_time,
                                    'batch_time': batch_time,
                                    'epoch': epoch,
                                    'step': steps
                                })
                                
                    except Exception as e:
                        print(f"Error in batch {batch_idx}: {str(e)}")
                        continue
                        
                # Track epoch time
                epoch_time = time.time() - epoch_start
                metrics['epoch_times'].append(epoch_time)
                print(f"\nEpoch {epoch} took {epoch_time:.2f}s")
                
            # Track total time
            metrics['total_time'] = time.time() - model_start
            print(f"\nTotal training time for {model_name}: {metrics['total_time']:.2f}s")
            
            # Compute and log timing statistics
            avg_batch = np.mean(metrics['batch_times'])
            avg_val = np.mean(metrics['validation_times'])
            print(f"Average batch time: {avg_batch:.3f}s")
            print(f"Average validation time: {avg_val:.3f}s")
            
        except Exception as e:
            print(f"Error in model {model_name}: {str(e)}")
            if self.config.wandb_logging:
                wandb.run.finish()
            
        return metrics

    def _validate_model(self, model: nn.Module) -> Tuple[float, float]:
        """Run validation and compute metrics"""
        model.eval()
        completed = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                try:
                    # Convert batch to device and handle target
                    batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                    if 'target' in batch:
                        batch['target_distribution'] = batch.pop('target')
                    
                    # Get model predictions
                    outputs = model(batch)
                    predictions = outputs.argmax(dim=1)
                    targets = batch['target_distribution'].argmax(dim=1)
                    
                    # Track accuracy
                    correct += (predictions == targets).sum().item()
                    
                    # Track completion rate
                    if 'word' in batch:
                        for i, word in enumerate(batch['word']):
                            pred_letter = chr(predictions[i].item() + ord('a'))
                            if pred_letter in word:
                                completed += 1
                        total += len(batch['word'])
                    
                except Exception as e:
                    print(f"Error in validation batch: {str(e)}")
                    print(f"Batch keys: {batch.keys()}")
                    continue
        
        completion_rate = completed / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        
        return completion_rate, accuracy

    def run_all_benchmarks(self) -> Dict[str, Dict[str, Any]]:
        """Run benchmarks for all model classes"""
        # Initialize data loaders if not already done
        if self.train_loader is None or self.val_loader is None:
            self.train_loader, self.val_loader = get_dataloaders(
                data_dir=self.data_dir,
                batch_size=self.config.batch_size,
                data_percentage=self.config.data_percentage,
                quick_mode=self.config.quick_mode,
                max_length=self.config.max_length
            )  # Removed p_value parameter

        # Initialize wandb if enabled
        if self.config.wandb_logging:
            try:
                wandb.init(
                    project="hangman-benchmarks",
                    config=self.config.__dict__,
                    name=f"benchmark-{time.strftime('%Y%m%d-%H%M%S')}"
                )
            except Exception as e:
                print(f"Failed to initialize wandb: {str(e)}")
                self.config.wandb_logging = False

        # Run benchmarks for each model
        for model_class in self.model_classes:
            model_name = model_class.__name__.lower()
            print(f"\nBenchmarking {model_name}...")
            
            # Get model-specific config
            model_config = self._get_model_config(model_class)
            
            try:
                metrics = self.train_model(
                    model_class=model_class,
                    model_name=model_name,
                    config=model_config
                )
                self.results[model_name] = metrics
                
            except Exception as e:
                print(f"Failed to benchmark {model_name}: {str(e)}")
                continue
                
        return self.results

    def _get_model_config(self, model_class: Type) -> Any:
        """Get appropriate configuration for model class"""
        base_config = {
            'input_dim': self.config.input_dim,
            'hidden_dim': self.config.hidden_dim,
            'max_length': self.config.max_length,
            'learning_rate': self.config.learning_rate,
            'weight_decay': self.config.weight_decay,
            'warmup_steps': self.config.warmup_steps
        }
        
        if model_class == TransformerModel:
            from hangman.models.transformer_model import TransformerConfig
            return TransformerConfig(**base_config)
        
        elif model_class == CNNLSTMModel:
            from hangman.models.cnn_lstm_model import CNNLSTMConfig
            return CNNLSTMConfig(**base_config)
        
        elif model_class == GNNModel:
            from hangman.models.gnn_model import GNNConfig
            return GNNConfig(**base_config)
        
        elif model_class == MLPAttentionModel:
            from hangman.models.mlp_attention_model import MLPAttentionConfig
            return MLPAttentionConfig(**base_config)
        
        else:
            raise ValueError(f"Unknown model class: {model_class.__name__}")

def main() -> None:
    """Main benchmark execution"""
    # Initialize configuration
    config = BenchmarkConfig()
    print(f"Running in {'QUICK' if config.quick_mode else 'FULL'} mode")
    
    # Initialize benchmarker
    benchmarker = ModelBenchmarker(
        config=config,
        model_classes=[TransformerModel, CNNLSTMModel, GNNModel, MLPAttentionModel],
        data_dir=Path("p_analysis")
    )
    
    # Run benchmarks
    try:
        benchmarker.run_all_benchmarks()
    except Exception as e:
        print(f"Benchmark failed: {str(e)}")
        raise
    finally:
        if config.wandb_logging:
            wandb.finish()

if __name__ == "__main__":
    main() 