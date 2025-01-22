import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from ..data.dataset import HangmanDataset

class BaseHangmanModel(nn.Module, ABC):
    """Base class for all Hangman models"""
    
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass returning letter probabilities"""
        pass
    
    def predict(self, state: Dict[str, torch.Tensor]) -> str:
        """Predict next letter given game state"""
        with torch.no_grad():
            logits = self(state)
            # Mask already guessed letters
            logits = logits.masked_fill(state['alphabet_state'].bool(), float('-inf'))
            # Return letter with highest probability
            return chr(logits.argmax().item() + ord('a'))

    def train_dataloader(self):
        """Return training dataloader"""
        if hasattr(self, 'train_words'):
            dataset = HangmanDataset(self.train_words)
            return DataLoader(
                dataset,
                batch_size=64,
                shuffle=True,
                num_workers=4
            )
        raise NotImplementedError("train_words not set")

    def val_dataloader(self):
        """Return validation dataloader"""
        if hasattr(self, 'val_words'):
            dataset = HangmanDataset(self.val_words)
            return DataLoader(
                dataset,
                batch_size=64,
                shuffle=False,
                num_workers=4
            )
        raise NotImplementedError("val_words not set") 