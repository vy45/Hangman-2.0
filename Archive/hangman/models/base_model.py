import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from ..data.dataset import HangmanDataset, HangmanValidationDataset
import torch.nn.functional as F
import wandb
import time
from dataclasses import dataclass
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

@dataclass
class ModelConfig:
    """Base configuration for all models"""
    input_dim: int = 28
    hidden_dim: int = 256
    max_length: int = 30
    dropout: float = 0.1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100

class BaseHangmanModel(pl.LightningModule):
    """Base class for all Hangman models"""
    
    def __init__(self, config: ModelConfig):
        """Initialize base model components
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Store configuration
        self.config = config
        
        # Initialize metrics
        self.train_steps: int = 0
        self.val_steps: int = 0
        self.best_val_loss: float = float('inf')
        
        # Initialize optimizers
        self.optimizer: Optional[Optimizer] = None
        self.scheduler: Optional[_LRScheduler] = None
        
    @abstractmethod
    def forward(self, word_state: torch.Tensor, alphabet_state: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model
        
        Args:
            word_state: [B, L, 28] tensor of word states
            alphabet_state: [B, 26] tensor of alphabet states
            
        Returns:
            [B, 26] tensor of letter probabilities
        """
        pass
    
    def predict(self, state: Dict[str, torch.Tensor]) -> str:
        """Predict next letter given game state"""
        with torch.no_grad():
            logits = self(state['word_state'], state['alphabet_state'])
            # Mask already guessed letters
            logits = logits.masked_fill(state['alphabet_state'].bool(), float('-inf'))
            # Return letter with highest probability
            return chr(logits.argmax().item() + ord('a'))

    def train_dataloader(self):
        """Return training dataloader"""
        dataset = HangmanDataset(self.train_data_path)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )

    def val_dataloader(self):
        """Return validation dataloader"""
        dataset = HangmanValidationDataset(self.val_words, self.word_processor)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize network weights
        
        Args:
            module: Neural network module to initialize
        """
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Common training step logic
        
        Args:
            batch: Input batch dictionary
            batch_idx: Index of current batch
            
        Returns:
            Loss tensor
        """
        # Track step
        self.train_steps += 1
        
        # Forward pass
        logits = self(batch['word_state'], batch['alphabet_state'])
        loss = nn.CrossEntropyLoss()(logits, batch['target'])
        
        # Log metrics
        self.log('train_loss', loss)
        accuracy = (logits.argmax(dim=1) == batch['target'].argmax(dim=1)).float().mean()
        self.log('train_accuracy', accuracy)
        
        return loss

    def on_train_epoch_end(self):
        # Confusion matrix
        wandb.log({
            "confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=self.all_targets,
                preds=self.all_preds,
                class_names=[chr(i + ord('a')) for i in range(26)]
            )
        })

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step with word completion tracking"""
        try:
            # Standard forward pass
            logits = self(batch)
            loss = F.cross_entropy(logits, batch['target_distribution'])
            
            # Track letter accuracy
            letter_accuracy = (logits.argmax(dim=1) == batch['target_distribution'].argmax(dim=1)).float().mean()
            
            # Track word completion rate (using target distribution)
            with torch.no_grad():
                batch_completion_rate = self._compute_completion_rate(
                    logits=logits,
                    target_words=batch['word'],
                    alphabet_state=batch['alphabet_state']
                )
            
            # Log metrics
            self.log('val_loss', loss, prog_bar=True)
            self.log('val_letter_accuracy', letter_accuracy, prog_bar=True)
            self.log('val_completion_rate', batch_completion_rate, prog_bar=True)
            
            return {
                'val_loss': loss,
                'val_accuracy': letter_accuracy,
                'completion_rate': batch_completion_rate
            }
        except Exception as e:
            print(f"Error in validation step: {str(e)}")
            print(f"Batch keys: {batch.keys()}")
            raise

    def _compute_completion_rate(
        self,
        logits: torch.Tensor,
        target_words: List[str],
        alphabet_state: torch.Tensor
    ) -> float:
        """Compute word completion rate using model predictions
        
        Args:
            logits: Model predictions [B, 26]
            target_words: List of target words
            alphabet_state: Current guessed letters [B, 26]
            
        Returns:
            Completion rate (0-1)
        """
        # Get predictions for unguessed letters
        probs = F.softmax(logits, dim=-1)
        probs = probs.masked_fill(alphabet_state.bool(), float('-inf'))
        predictions = probs.argmax(dim=-1)
        
        completed = 0
        total = len(target_words)
        
        for i, word in enumerate(target_words):
            # Check if predicted letter is in word
            pred_letter = chr(predictions[i].item() + ord('a'))
            if pred_letter in word:
                completed += 1
            
        return completed / total