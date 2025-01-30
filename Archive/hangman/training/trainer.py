import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List
from tqdm import tqdm
import matplotlib.pyplot as plt
from ..data.simulator import HangmanSimulator
from ..models.base_model import BaseHangmanModel

class HangmanTrainer:
    def __init__(
        self,
        model: BaseHangmanModel,
        simulator: HangmanSimulator,
        device: torch.device,
        batch_size: int = 64,
        learning_rate: float = 0.001
    ):
        self.model = model.to(device)
        self.simulator = simulator
        self.device = device
        self.batch_size = batch_size
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
    def _prepare_batch(self, states: List[Dict]) -> Dict[str, torch.Tensor]:
        """Convert batch of states to tensors"""
        return {
            'word_state': torch.FloatTensor([s['word_state'] for s in states]).to(self.device),
            'alphabet_state': torch.FloatTensor([s['alphabet_state'] for s in states]).to(self.device),
            'word_length': torch.LongTensor([s['word_length'] for s in states]).to(self.device),
            'target_distribution': torch.FloatTensor([s['target_distribution'] for s in states]).to(self.device)
        }
    
    def train_epoch(self, train_words: List[str], steps: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(range(steps), desc='Training')
        for _ in pbar:
            # Generate batch of training data
            batch_states = []
            for _ in range(self.batch_size):
                word = random.choice(train_words)
                states = self.simulator.simulate_game(word)
                if states:  # Ensure we have valid states
                    batch_states.append(random.choice(states))
            
            # Prepare batch
            batch = self._prepare_batch(batch_states)
            
            # Forward pass
            logits = self.model(batch)
            loss = self.criterion(logits, batch['target_distribution'])
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        return total_loss / steps
    
    def validate(self, val_words: List[str], steps: int) -> Dict[str, float]:
        """Validate model performance"""
        self.model.eval()
        total_loss = 0
        correct_guesses = 0
        total_guesses = 0
        
        with torch.no_grad():
            for _ in range(steps):
                # Generate batch of validation data
                batch_states = []
                for _ in range(self.batch_size):
                    word = random.choice(val_words)
                    states = self.simulator.simulate_game(word)
                    if states:
                        batch_states.append(random.choice(states))
                
                # Prepare batch
                batch = self._prepare_batch(batch_states)
                
                # Forward pass
                logits = self.model(batch)
                loss = self.criterion(logits, batch['target_distribution'])
                
                # Calculate accuracy
                pred = logits.argmax(dim=1)
                target = batch['target_distribution'].argmax(dim=1)
                correct_guesses += (pred == target).sum().item()
                total_guesses += pred.size(0)
                
                total_loss += loss.item()
        
        return {
            'loss': total_loss / steps,
            'accuracy': correct_guesses / total_guesses
        } 