import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

class HangmanSupervisedDataset:
    def __init__(self, words, max_word_length=30):
        self.words = words
        self.max_word_length = max_word_length
        
    def generate_training_sample(self, word):
        """Generate a training sample from a word"""
        word = word.lower()
        # Initialize random number of previous guesses
        num_prev_guesses = random.randint(0, len(word) - 1)
        guessed_letters = set()
        
        # Randomly select some letters to be "previously guessed"
        word_letters = set(word)
        available_letters = set('abcdefghijklmnopqrstuvwxyz') - word_letters
        
        # Add some correct and incorrect guesses
        num_correct = random.randint(0, min(num_prev_guesses, len(word_letters)))
        num_incorrect = num_prev_guesses - num_correct
        
        if num_correct > 0:
            guessed_letters.update(random.sample(word_letters, num_correct))
        if num_incorrect > 0:
            guessed_letters.update(random.sample(list(available_letters), num_incorrect))
            
        # Create current state representation
        current_state = ['_'] * len(word)
        for i, letter in enumerate(word):
            if letter in guessed_letters:
                current_state[i] = letter
                
        # Get next correct letter (target)
        remaining_letters = set(word) - guessed_letters
        if not remaining_letters:
            return None  # Skip if no remaining letters
        
        target_letter = random.choice(list(remaining_letters))
        
        return self._create_feature_vector(current_state, guessed_letters), self._letter_to_index(target_letter)
    
    def _create_feature_vector(self, current_state, guessed_letters):
        """Create feature vector from game state"""
        # One-hot encoding for word state (max_length × 27)
        word_state = np.zeros((self.max_word_length, 27))
        for i, char in enumerate(current_state):
            if char == '_':
                word_state[i, 26] = 1  # mask token
            else:
                word_state[i, ord(char) - ord('a')] = 1
        
        # Pad remaining positions
        for i in range(len(current_state), self.max_word_length):
            word_state[i, 26] = 1
            
        # One-hot encoding for guessed letters (26)
        alphabet_state = np.zeros(26)
        for letter in guessed_letters:
            alphabet_state[ord(letter) - ord('a')] = 1
            
        return {
            'word_state': word_state,
            'alphabet_state': alphabet_state,
            'word_length': len(current_state)
        }
    
    def _letter_to_index(self, letter):
        """Convert letter to index"""
        return ord(letter) - ord('a')
    
    def generate_batch(self, batch_size):
        """Generate a batch of training samples"""
        batch_features = []
        batch_targets = []
        
        while len(batch_features) < batch_size:
            word = random.choice(self.words)
            sample = self.generate_training_sample(word)
            if sample is not None:
                features, target = sample
                batch_features.append(features)
                batch_targets.append(target)
        
        return self._collate_batch(batch_features, batch_targets)
    
    def _collate_batch(self, features, targets):
        """Collate batch into tensor format"""
        return {
            'word_state': torch.FloatTensor([f['word_state'] for f in features]),
            'alphabet_state': torch.FloatTensor([f['alphabet_state'] for f in features]),
            'word_length': torch.LongTensor([f['word_length'] for f in features])
        }, torch.LongTensor(targets)

class HangmanSupervisedModel(nn.Module):
    def __init__(self, max_word_length=30):
        super(HangmanSupervisedModel, self).__init__()
        
        # Word processing
        self.conv = nn.Conv1d(27, 32, kernel_size=3, padding=1)
        self.batch_norm = nn.BatchNorm1d(32)
        
        # LSTM for sequence processing
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            dropout=0.1
        )
        
        # Combination layers
        self.combine_features = nn.Sequential(
            nn.Linear(64 + 26, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 26)
        )
    
    def forward(self, x):
        # Process word state
        word_state = x['word_state'].transpose(1, 2)
        conv_out = self.conv(word_state)
        conv_out = self.batch_norm(conv_out)
        conv_out = torch.relu(conv_out)
        conv_out = conv_out.transpose(1, 2)
        
        # LSTM processing
        try:
            packed_sequence = nn.utils.rnn.pack_padded_sequence(
                conv_out,
                x['word_length'].cpu(),
                batch_first=True,
                enforce_sorted=False
            )
            
            self.lstm.flatten_parameters()
            lstm_out, _ = self.lstm(packed_sequence)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
            
        except RuntimeError:
            lstm_out = self.lstm(conv_out)[0]
        
        # Get last relevant output for each sequence
        batch_size = lstm_out.size(0)
        lstm_features = torch.zeros(batch_size, 64, device=lstm_out.device)
        for i in range(batch_size):
            lstm_features[i] = lstm_out[i, x['word_length'][i] - 1]
        
        # Combine with alphabet state
        combined = torch.cat([lstm_features, x['alphabet_state']], dim=1)
        
        # Get letter probabilities
        logits = self.combine_features(combined)
        
        # Mask already guessed letters
        mask = (x['alphabet_state'] == 1)
        logits = logits.masked_fill(mask, float('-inf'))
        
        return logits

def train_supervised_model(train_words, val_words, num_epochs=10, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = HangmanSupervisedDataset(train_words)
    val_dataset = HangmanSupervisedDataset(val_words)
    
    # Create model
    model = HangmanSupervisedModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    steps_per_epoch = 1000
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        
        # Training steps
        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{num_epochs}")
        for _ in pbar:
            features, targets = train_dataset.generate_batch(batch_size)
            
            # Move to device
            features = {k: v.to(device) for k, v in features.items()}
            targets = targets.to(device)
            
            # Forward pass
            logits = model(features)
            loss = criterion(logits, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            pbar.set_postfix({'loss': f"{np.mean(epoch_losses):.4f}"})
        
        train_losses.append(np.mean(epoch_losses))
        
        # Validation
        model.eval()
        val_epoch_losses = []
        with torch.no_grad():
            for _ in range(100):  # Validate on 100 batches
                features, targets = val_dataset.generate_batch(batch_size)
                features = {k: v.to(device) for k, v in features.items()}
                targets = targets.to(device)
                
                logits = model(features)
                loss = criterion(logits, targets)
                val_epoch_losses.append(loss.item())
        
        val_loss = np.mean(val_epoch_losses)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1} - Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_loss:.4f}")
        
        # Plot training progress
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.title('Training Progress')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
    
    return model

# Usage example:
if __name__ == "__main__":
    # Load words
    with open('words_250000_train.txt', 'r') as f:
        word_list = [word.strip().lower() for word in f.readlines()]
    
    # Split into train and validation sets
    random.shuffle(word_list)
    split_idx = int(len(word_list) * 0.8)
    train_words = word_list[:split_idx]
    val_words = word_list[split_idx:]
    
    # Train model
    model = train_supervised_model(train_words, val_words) 