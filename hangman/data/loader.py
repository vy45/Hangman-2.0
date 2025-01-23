from typing import Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from pathlib import Path
from dataclasses import dataclass
import time
from tqdm import tqdm

@dataclass
class DataLoaderConfig:
    """Configuration for data loading"""
    batch_size: int = 32
    num_workers: int = 4
    max_length: int = 30
    pin_memory: bool = True
    persistent_workers: bool = True
    shuffle_train: bool = True
    quick_mode: bool = False
    data_percentage: float = 100.0
    cache_size: int = 4

class HangmanDataLoader:
    """Unified data loading interface for Hangman
    
    This class handles:
    1. Dataset initialization
    2. Batch creation
    3. Memory management
    4. Data sampling
    5. Performance optimization
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        config: DataLoaderConfig,
        split: str = 'train'
    ):
        """Initialize data loader
        
        Args:
            data_dir: Data directory
            config: Loader configuration
            split: 'train' or 'val'
        """
        self.data_dir = Path(data_dir)
        self.config = config
        self.split = split
        
        # Validate setup
        self._validate_setup()
        
        # Initialize datasets
        self.dataset = self._create_dataset()
        
        # Sample data if needed
        if config.data_percentage < 100 or config.quick_mode:
            self.dataset = self._sample_dataset()
            
        # Create dataloader
        self.dataloader = self._create_dataloader()
        
    def _validate_setup(self) -> None:
        """Validate configuration and paths"""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
            
        if self.split not in {'train', 'val'}:
            raise ValueError(f"Invalid split: {self.split}")
    
    def _create_dataset(self) -> Dataset:
        """Create appropriate dataset based on split"""
        if self.split == 'train':
            return HangmanDataset(
                self.data_dir,
                max_length=self.config.max_length
            )
        else:
            return HangmanValidationDataset(
                self.data_dir,
                max_length=self.config.max_length
            )
            
    def _sample_dataset(self) -> Dataset:
        """Sample subset of dataset if needed"""
        full_size = len(self.dataset)
        
        # Calculate sample size
        if self.config.quick_mode:
            sample_size = full_size // 20  # 5% for quick mode
        else:
            sample_size = int(full_size * self.config.data_percentage / 100)
            
        # Create subset
        indices = torch.randperm(full_size)[:sample_size]
        return Subset(self.dataset, indices)
    
    def _create_dataloader(self) -> DataLoader:
        """Create DataLoader with proper configuration"""
        return DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers if not self.config.quick_mode else 0,
            pin_memory=self.config.pin_memory and torch.cuda.is_available(),
            persistent_workers=self.config.persistent_workers and not self.config.quick_mode,
            shuffle=self.config.shuffle_train and self.split == 'train'
        ) 