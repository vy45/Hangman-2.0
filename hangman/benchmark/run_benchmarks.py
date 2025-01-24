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
    log_every_n_steps: int = 50 if quick_mode else 50
    validation_size: Optional[int] = None  # Will be set to 5% in quick mode
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
            self.max_epochs = 1
            self.patience = 2
            self.log_every_n_steps = 50
            # Set validation size to 5% of total validation words (45473)
            self.validation_size = int(45473 * 0.05)  # ~2274 words
        else:
            self.validation_size = None  # Use full validation set
        
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
            'epoch_times': [],
            'batch_times': [],
            'total_time': 0.0
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
            total_steps = len(self.train_loader) * self.config.max_epochs
            steps = 0
            
            for epoch in range(self.config.max_epochs):
                if steps >= total_steps:
                    break
                    
                epoch_start = time.time()
                model.train()
                
                for batch_idx, batch in enumerate(tqdm(self.train_loader, 
                                                     desc=f"Epoch {epoch}/{self.config.max_epochs-1}")):
                    if steps >= total_steps:
                        break
                        
                    batch_start = time.time()
                    try:
                        batch = {k: v.to(self.device) for k, v in batch.items()}
                        batch['target_distribution'] = batch.pop('target')
                        
                        outputs = model(batch)
                        loss = criterion(outputs, batch['target_distribution'])
                        
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                        batch_time = time.time() - batch_start
                        metrics['batch_times'].append(batch_time)
                        
                        steps += 1
                        if steps % self.config.log_every_n_steps == 0:
                            metrics['train_loss'].append(loss.item())
                            
                            try:
                                # Time validation
                                print(f"\nStarting validation at step {steps}...")
                                val_start = time.time()
                                completion_rate, val_accuracy = self._validate_model(model)
                                val_time = time.time() - val_start
                                
                                metrics['completion_rate'].append(completion_rate)
                                metrics['val_accuracy'].append(val_accuracy)
                                metrics['validation_times'].append(val_time)
                                
                                print(f"Step {steps} metrics:")
                                print(f"Batch Time: {batch_time:.3f}s")
                                print(f"Validation Time: {val_time:.2f}s ({val_time/60:.1f}min)")
                                print(f"Completion Rate: {completion_rate:.3f}")
                                print(f"Validation Accuracy: {val_accuracy:.3f}")
                                print(f"Average validation time: {np.mean(metrics['validation_times']):.2f}s")
                                
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
                                print(f"\nValidation failed at step {steps}")
                                print("Training stopped.")
                                raise  # Stop training
                            
                    except Exception as e:
                        print(f"Error in batch {batch_idx}: {str(e)}")
                        continue
                
                # Track epoch time
                epoch_time = time.time() - epoch_start
                metrics['epoch_times'].append(epoch_time)
                print(f"\nEpoch {epoch}/{self.config.max_epochs-1} took {epoch_time:.2f}s")
                
                if steps >= total_steps:
                    break
            
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
        """Run validation by playing complete Hangman games."""
        model.eval()
        completed_words = 0
        correct_guesses = 0
        total_words = 0
        
        # Track memory usage
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            start_mem = torch.cuda.memory_allocated()
        
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            for batch_idx, batch in enumerate(self.val_loader):
                if batch_idx % 100 == 0:  # Print progress
                    print(f"Validating batch {batch_idx}/{len(self.val_loader)}")
                    if torch.cuda.is_available():
                        current_mem = torch.cuda.memory_allocated()
                        peak_mem = torch.cuda.max_memory_allocated()
                        print(f"Current GPU memory: {current_mem/1e9:.2f}GB")
                        print(f"Peak GPU memory: {peak_mem/1e9:.2f}GB")
                
                try:
                    # Move tensors to device and get word
                    tensor_batch = {k: v.to(self.device) for k, v in batch.items() 
                                  if isinstance(v, torch.Tensor) and k != 'word'}
                    words = batch['word']
                    
                    # Play complete games for each word in batch
                    for i, word in enumerate(words):
                        word = word.lower()
                        guessed_letters = set()
                        wrong_guesses = 0
                        word_letters = set(word)
                        current_batch = {k: v[i:i+1] for k, v in tensor_batch.items()}
                        
                        # Play until word is complete or too many wrong guesses
                        while wrong_guesses < 6 and len(word_letters - guessed_letters) > 0:
                            # Get model's guess
                            outputs = model(current_batch)
                            pred_probs = torch.softmax(outputs[0], dim=0)
                            
                            # Clear unnecessary tensors
                            del outputs
                            
                            # Filter out already guessed letters
                            for g in guessed_letters:
                                pred_probs[ord(g) - ord('a')] = 0
                            
                            # Get prediction
                            pred_letter = chr(pred_probs.argmax().item() + ord('a'))
                            guessed_letters.add(pred_letter)
                            
                            # Clear more tensors
                            del pred_probs
                            
                            # Check if correct
                            if pred_letter in word:
                                correct_guesses += word.count(pred_letter)
                            else:
                                wrong_guesses += 1
                            
                            # Update game state for next guess
                            current_batch = self._update_game_state(current_batch, pred_letter, word)
                        
                        # Check if word was completed
                        if len(word_letters - guessed_letters) == 0 and wrong_guesses < 6:
                            completed_words += 1
                        
                        total_words += 1
                        
                        # Clear batch tensors
                        del current_batch
                    
                    # Clear main batch tensors
                    del tensor_batch
                    
                except Exception as e:
                    print(f"Error in validation batch {batch_idx}:")
                    print(f"Error: {str(e)}")
                    raise
                
                # Force GPU memory cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        if total_words == 0:
            raise ValueError("No words were processed in validation")
        
        # Print final memory stats
        if torch.cuda.is_available():
            end_mem = torch.cuda.memory_allocated()
            peak_mem = torch.cuda.max_memory_allocated()
            print(f"\nValidation memory usage:")
            print(f"Start: {start_mem/1e9:.2f}GB")
            print(f"End: {end_mem/1e9:.2f}GB")
            print(f"Peak: {peak_mem/1e9:.2f}GB")
        
        completion_rate = completed_words / total_words
        accuracy = correct_guesses / total_words
        
        return completion_rate, accuracy

    def _update_game_state(self, batch: Dict[str, torch.Tensor], guessed_letter: str, word: str) -> Dict[str, torch.Tensor]:
        """Update game state after a guess."""
        # Update word state while preserving padding
        word_state = batch['word_state'].clone()
        word_length = len(word)
        
        # Only update up to the actual word length
        for i in range(word_length):
            if word[i] == guessed_letter:
                # Set one-hot encoding for the guessed letter
                word_state[0, i, :26] = 0  # Clear letter positions
                word_state[0, i, 26] = 0   # Clear mask position
                word_state[0, i, ord(guessed_letter) - ord('a')] = 1
        
        # Update alphabet state
        alphabet_state = batch['alphabet_state'].clone()
        alphabet_state[0, ord(guessed_letter) - ord('a')] = 0  # Mark letter as guessed
        
        return {
            'word_state': word_state,
            'attention_mask': batch['attention_mask'],  # Keep original attention mask
            'alphabet_state': alphabet_state,
            'word_length': batch['word_length']
        }

    def run_all_benchmarks(self) -> Dict[str, Dict[str, Any]]:
        """Run benchmarks for all model classes"""
        # Initialize data loaders if not already done
        if self.train_loader is None or self.val_loader is None:
            print(f"Initializing dataloaders with validation_size: {self.config.validation_size}")  # Debug print
            self.train_loader, self.val_loader = get_dataloaders(
                data_dir=self.data_dir,
                batch_size=self.config.batch_size,
                data_percentage=self.config.data_percentage,
                quick_mode=self.config.quick_mode,
                max_length=self.config.max_length,
                validation_size=self.config.validation_size
            )

        # Run benchmarks for each model
        for model_class in self.model_classes:
            model_name = model_class.__name__.lower()
            print(f"\nBenchmarking {model_name}...")
            
            # Initialize wandb for this model if enabled
            if self.config.wandb_logging:
                try:
                    wandb.init(
                        project="hangman-benchmarks",
                        name=f"{model_name}-{time.strftime('%Y%m%d-%H%M%S')}",
                        config={
                            **self.config.__dict__,
                            'model': model_name,
                            'n_params': sum(p.numel() for p in model_class(self._get_model_config(model_class)).parameters())
                        },
                        reinit=True  # Allow multiple runs
                    )
                except Exception as e:
                    print(f"Failed to initialize wandb for {model_name}: {str(e)}")
                    self.config.wandb_logging = False
            
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
            finally:
                if self.config.wandb_logging:
                    wandb.finish()  # Close this model's run
                
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