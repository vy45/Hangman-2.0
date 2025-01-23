import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import glob
from typing import List, Dict, Optional, Tuple
from .word_processor import WordProcessor
from .simulator import HangmanSimulator
from functools import lru_cache
import time
from datetime import datetime
from tqdm import tqdm
import os
from torch.utils.data import DataLoader, Subset


class BaseHangmanDataset(Dataset):
    """Base class for Hangman datasets"""
    def __init__(self):
        super().__init__()
    
    def __len__(self):
        raise NotImplementedError
        
    def __getitem__(self, idx):
        raise NotImplementedError


class HangmanDataset(Dataset):
    """Dataset for loading and preprocessing Hangman game states
    
    This class handles:
    1. Loading state batches efficiently
    2. Padding sequences to fixed length
    3. Creating attention masks
    4. Caching frequently accessed batches
    
    Attributes:
        max_length (int): Maximum sequence length for padding
        data_dir (Path): Directory containing state batches
        batch_files (List[str]): List of batch file paths
        total_states (int): Total number of game states
        batch_offsets (List[int]): Cumulative state counts for batch indexing
    """
    
    def __init__(self, 
                 data_dir: str, 
                 p_value: float = 0.5, 
                 max_length: int = 30,
                 quick_mode: bool = False):
        """Initialize dataset
        
        Args:
            data_dir: Base directory containing p_value folders
            p_value: Which p-value dataset to use (0.5 = balanced)
            max_length: Maximum sequence length for padding
            quick_mode: If True, only load a subset of batches for fast testing
        """
        load_start = time.time()
        print(f"\nInitializing dataset from {data_dir} at {datetime.now().strftime('%H:%M:%S')}")
        
        self.max_length = max_length
        self.data_dir = Path(data_dir) / f"p_{int(p_value*100)}" / "states"
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset not found: {self.data_dir}")
        
        # Get and verify batch files
        self.batch_files = sorted(glob.glob(str(self.data_dir / "batch_*.npz")))
        if not self.batch_files:
            raise FileNotFoundError(f"No batch files found in {self.data_dir}")
            
        # Quick mode: use subset of batches
        if quick_mode:
            num_batches = max(1, len(self.batch_files) // 20)  # Use 5% of batches
            self.batch_files = self.batch_files[:num_batches]
            print(f"Quick mode: Using {num_batches} batches")
        
        print(f"Found {len(self.batch_files)} batch files in {time.time() - load_start:.2f}s")
        
        # Index all batches
        self._index_batches()
    
    def _index_batches(self):
        """Create index mapping state indices to batch files"""
        index_start = time.time()
        self.total_states = 0
        self.batch_offsets = [0]
        
        for batch_file in tqdm(self.batch_files, desc="Indexing batches"):
            with np.load(batch_file, allow_pickle=True) as data:
                size = len(data['word_states'])
                self.total_states += size
                self.batch_offsets.append(self.total_states)
        
        print(f"Indexed {self.total_states:,} states in {time.time() - index_start:.2f}s")
    
    def __len__(self) -> int:
        return self.total_states
    
    @lru_cache(maxsize=4)
    def _load_batch(self, batch_idx: int) -> Dict:
        """Load and cache batch data
        
        Uses LRU caching to keep most recently used batches in memory
        """
        with np.load(self.batch_files[batch_idx], allow_pickle=True) as data:
            return {k: data[k].copy() for k in data.files}  # Copy to make writable
    
    def _pad_sequence(self, 
                     sequence: np.ndarray, 
                     max_length: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad sequence and create attention mask
        
        Args:
            sequence: Input sequence [seq_len, features]
            max_length: Target length for padding (default: self.max_length)
            
        Returns:
            tuple: (padded_sequence, attention_mask)
        """
        max_length = max_length or self.max_length
        seq_len = sequence.shape[0]
        
        # Create padded sequence
        padded = np.zeros((max_length, sequence.shape[1]), dtype=sequence.dtype)
        padded[:seq_len] = sequence
        
        # Create attention mask (1 for real tokens, 0 for padding)
        mask = np.zeros(max_length, dtype=np.bool_)
        mask[:seq_len] = 1
        
        return torch.FloatTensor(padded), torch.BoolTensor(mask)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single game state with padding and masking
        
        Args:
            idx: State index
            
        Returns:
            dict containing:
                - word_state: Padded word state [max_len, features]
                - attention_mask: Boolean mask [max_len]
                - alphabet_state: Available letters [26]
                - word_length: Original sequence length [1]
                - target: Target distribution [26]
        """
        # Find batch using binary search
        batch_idx = np.searchsorted(self.batch_offsets, idx, side='right') - 1
        local_idx = idx - self.batch_offsets[batch_idx]
        
        # Load batch data
        batch = self._load_batch(batch_idx)
        word_state = np.frombuffer(batch['word_states'][local_idx]).reshape(batch['word_shapes'][local_idx])
        
        # Pad sequence and create mask
        padded_state, attention_mask = self._pad_sequence(word_state)
        
        return {
            'word_state': padded_state,
            'attention_mask': attention_mask,
            'alphabet_state': torch.FloatTensor(batch['alphabet_states'][local_idx]),
            'word_length': torch.LongTensor([batch['word_lengths'][local_idx]]),
            'target': torch.FloatTensor(batch['targets'][local_idx])
        }


class HangmanValidationDataset(BaseHangmanDataset):
    """Dataset for validation using real games on unseen words"""
    
    def __init__(self, data_dir: str, word_processor: WordProcessor, max_length: int = 30):
        """Initialize validation dataset
        
        Args:
            data_dir: Base directory containing validation_words.txt
            word_processor: WordProcessor instance for word handling
            max_length: Maximum sequence length for padding (default: 30)
        """
        super().__init__()
        self.max_length = max_length
        
        val_file = Path(data_dir) / "validation_words.txt"
        if not val_file.exists():
            raise FileNotFoundError(f"Validation file not found: {val_file}")
            
        with open(val_file, 'r') as f:
            self.val_words = [w.strip() for w in f.readlines()]
            
        self.word_processor = word_processor
        self.simulator = HangmanSimulator(word_processor, optimal_prob=1.0)
        print(f"Loaded validation dataset with {len(self.val_words):,} words")
    
    def __len__(self):
        return len(self.val_words)
    
    def __getitem__(self, idx):
        word = self.val_words[idx]
        # Get initial state (no letters guessed)
        state = self.simulator.generate_game_state(word, set())
        
        # Pad sequence and create mask
        word_state = state['word_state']
        padded_state, attention_mask = self._pad_sequence(word_state)
        
        return {
            'word_state': padded_state,
            'attention_mask': attention_mask,
            'alphabet_state': torch.FloatTensor(state['alphabet_state']),
            'word_length': torch.LongTensor([state['word_length']]),
            'target': torch.FloatTensor(state['target_distribution']),
            'word': word  # Make sure this is included!
        }

    def _pad_sequence(self, sequence, max_length=None):
        """Same padding logic as HangmanDataset"""
        max_length = max_length or self.max_length
        seq_len = sequence.shape[0]
        
        padded = np.zeros((max_length, sequence.shape[1]), dtype=sequence.dtype)
        padded[:seq_len] = sequence
        
        mask = np.zeros(max_length, dtype=np.bool_)
        mask[:seq_len] = 1
        
        return torch.FloatTensor(padded), torch.BoolTensor(mask)


def get_dataloaders(
    data_dir: Path,
    batch_size: int,
    data_percentage: float = 100.0,
    quick_mode: bool = False,
    max_length: int = 30
) -> Tuple[DataLoader, DataLoader]:
    """Get train and validation dataloaders"""
    print("Creating dataloaders...")
    print(f"Data directory: {data_dir}")
    print(f"Quick mode: {quick_mode}")
    print(f"Data percentage: {data_percentage}%")
    
    # Create datasets
    train_dataset = HangmanDataset(
        data_dir=str(data_dir),
        p_value=0.5,
        max_length=max_length,
        quick_mode=quick_mode
    )
    
    # Load validation words first
    val_file = Path(data_dir) / "validation_words.txt"
    with open(val_file, 'r') as f:
        val_words = [w.strip() for w in f.readlines()]
    
    # Create validation dataset with proper word processor
    val_dataset = HangmanValidationDataset(
        data_dir=str(data_dir),
        word_processor=WordProcessor(val_words),  # Pass the actual validation words
        max_length=max_length
    )
    
    # Sample training data if needed
    if data_percentage < 100:
        num_train = int(len(train_dataset) * data_percentage / 100)
        indices = torch.randperm(len(train_dataset))[:num_train]
        train_dataset = Subset(train_dataset, indices)
        print(f"Using {data_percentage}% of training data: {num_train:,} samples")
    
    if quick_mode:
        num_val = len(val_dataset) // 5
        val_indices = torch.randperm(len(val_dataset))[:num_val]
        val_dataset = Subset(val_dataset, val_indices)
        print(f"Quick mode: Using {num_val} validation samples")
    
    # Configure dataloaders
    dataloader_kwargs = {
        'batch_size': batch_size,
        'num_workers': 0 if quick_mode else 4,
        'pin_memory': torch.cuda.is_available(),
        'persistent_workers': False if quick_mode else True
    }
    
    train_loader = DataLoader(train_dataset, shuffle=True, **dataloader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **dataloader_kwargs)
    
    return train_loader, val_loader 