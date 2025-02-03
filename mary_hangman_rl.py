import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict, deque
import random
from tqdm import tqdm
import logging
from pathlib import Path
from datetime import datetime
import sys
import pickle
import string
from mary_hangman_nv3 import MaryLSTMModel  # Only import the model class

# Setup logging
def setup_logging():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f'mary_hangman_rl_{timestamp}.log'
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_file

def length_to_binary(length, num_bits=5):
    """Convert length to binary representation"""
    binary = format(min(length, 2**num_bits - 1), f'0{num_bits}b')
    return [int(b) for b in binary]

def prepare_input(state, device):
    """Convert state to model input tensors with proper dimensions"""
    try:
        # Convert current state to char indices (0 = padding, 1-26 = letters, 27 = '_')
        char_indices = torch.zeros(32, dtype=torch.long, device=device)
        for i, char in enumerate(state['current_state']):
            if char == '_':
                char_indices[i] = 27
            else:
                char_indices[i] = ord(char.lower()) - ord('a') + 1
                
        # Create guessed letters tensor (one-hot)
        guessed = torch.zeros(26, device=device)
        for letter in state['guessed_letters']:
            guessed[ord(letter) - ord('a')] = 1
        
        # Convert word length to 5-bit binary
        word_length_bits = torch.tensor(
            length_to_binary(len(state['current_state'])), 
            dtype=torch.float32, 
            device=device
        )
        
        # Other features
        vowel_ratio = torch.tensor(state['vowel_ratio'], dtype=torch.float32, device=device)
        remaining_lives = torch.tensor(state['remaining_lives'], dtype=torch.float32, device=device)
        
        return char_indices, guessed, word_length_bits, vowel_ratio, remaining_lives
        
    except Exception as e:
        logging.error(f"Error in prepare_input: {str(e)}")
        logging.error(f"State: {state}")
        raise

class HangmanDQN(nn.Module):
    def __init__(self, char_embedding_dim=32, hidden_dim=128):
        super().__init__()
        
        # Character embedding (28 = 26 letters + '_' + padding)
        self.char_embedding = nn.Embedding(28, char_embedding_dim, padding_idx=0)
        
        # LSTM for processing character sequence
        self.lstm = nn.LSTM(
            input_size=char_embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        
        # Additional features size: 
        # 26 (guessed) + 5 (word_length_bits) + 1 (vowel_ratio) + 1 (lives)
        additional_features = 33
        
        # Combine LSTM output with additional features
        self.network = nn.Sequential(
            nn.Linear(hidden_dim + additional_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 26)  # Q-value for each letter
        )
    
    def forward(self, char_indices, guessed, word_length_bits, vowel_ratio, lives):
        batch_size = char_indices.size(0)
        
        # Log input shapes
        logging.debug(f"\nInput shapes:")
        logging.debug(f"char_indices: {char_indices.shape}")
        logging.debug(f"guessed: {guessed.shape}")
        logging.debug(f"word_length_bits: {word_length_bits.shape}")
        logging.debug(f"vowel_ratio: {vowel_ratio.shape}")
        logging.debug(f"lives: {lives.shape}")
        
        # Embed characters
        embedded = self.char_embedding(char_indices)  # [batch_size, seq_len, embedding_dim]
        logging.debug(f"After embedding: {embedded.shape}")
        
        # Process through LSTM
        lstm_out, _ = self.lstm(embedded)  # [batch_size, seq_len, hidden_dim]
        lstm_features = lstm_out[:, -1]  # Take last LSTM output [batch_size, hidden_dim]
        logging.debug(f"LSTM features: {lstm_features.shape}")
        
        # Make sure all tensors have correct batch dimension
        if guessed.dim() == 1:
            guessed = guessed.unsqueeze(0).expand(batch_size, -1)
        elif guessed.size(0) == 1:
            guessed = guessed.expand(batch_size, -1)
        
        if word_length_bits.dim() == 1:
            word_length_bits = word_length_bits.unsqueeze(0).expand(batch_size, -1)
        elif word_length_bits.size(0) == 1:
            word_length_bits = word_length_bits.expand(batch_size, -1)
        
        if vowel_ratio.dim() == 0:
            vowel_ratio = vowel_ratio.view(1, 1).expand(batch_size, -1)
        elif vowel_ratio.dim() == 1:
            vowel_ratio = vowel_ratio.unsqueeze(1).expand(batch_size, -1)
        elif vowel_ratio.size(0) == 1:
            vowel_ratio = vowel_ratio.expand(batch_size, -1)
        
        if lives.dim() == 0:
            lives = lives.view(1, 1).expand(batch_size, -1)
        elif lives.dim() == 1:
            lives = lives.unsqueeze(1).expand(batch_size, -1)
        elif lives.size(0) == 1:
            lives = lives.expand(batch_size, -1)
        
        # Log shapes after adjustments
        logging.debug(f"\nShapes before concatenation:")
        logging.debug(f"lstm_features: {lstm_features.shape}")
        logging.debug(f"guessed: {guessed.shape}")
        logging.debug(f"word_length_bits: {word_length_bits.shape}")
        logging.debug(f"vowel_ratio: {vowel_ratio.shape}")
        logging.debug(f"lives: {lives.shape}")
        
        try:
            # Concatenate with additional features
            features = torch.cat([
                lstm_features,      # [batch_size, hidden_dim]
                guessed,           # [batch_size, 26]
                word_length_bits,  # [batch_size, 5]
                vowel_ratio,       # [batch_size, 1]
                lives             # [batch_size, 1]
            ], dim=1)
            
            logging.debug(f"After concatenation: {features.shape}")
            
            # Get Q-values
            q_values = self.network(features)
            logging.debug(f"Q-values shape: {q_values.shape}")
            
            return q_values
            
        except Exception as e:
            logging.error("Error during tensor concatenation:")
            logging.error(f"lstm_features: {lstm_features.shape}, {lstm_features.dtype}")
            logging.error(f"guessed: {guessed.shape}, {guessed.dtype}")
            logging.error(f"word_length_bits: {word_length_bits.shape}, {word_length_bits.dtype}")
            logging.error(f"vowel_ratio: {vowel_ratio.shape}, {vowel_ratio.dtype}")
            logging.error(f"lives: {lives.shape}, {lives.dtype}")
            raise

class HangmanAgent:
    def __init__(self, device='cpu', lr=1e-4):
        self.device = device
        self.gamma = 0.99
        self.batch_size = 32
        self.target_update = 1000  # Update target network every N steps
        self.steps = 0
        
        # Q Networks
        self.policy_net = HangmanDQN().to(device)
        self.target_net = HangmanDQN().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = deque(maxlen=100000)
        
        # Metrics
        self.training_wins = []
        self.training_losses = []
    
    def get_action(self, state, epsilon):
        if random.random() < epsilon:
            # Random action from unguessed letters
            valid_actions = [i for i in range(26) 
                           if chr(i + ord('a')) not in state['guessed_letters']]
            return chr(random.choice(valid_actions) + ord('a'))
        
        with torch.no_grad():
            char_indices, guessed, word_length_bits, vowel_ratio, lives = prepare_input(state, self.device)
            q_values = self.policy_net(char_indices, guessed, word_length_bits, vowel_ratio, lives)
            
            # Mask guessed letters
            for letter in state['guessed_letters']:
                q_values[0, ord(letter) - ord('a')] = float('-inf')
            
            return chr(q_values.argmax().item() + ord('a'))
    
    def train_step(self, batch):
        if len(batch) < self.batch_size:
            return 0
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Process current states
        current_q_values = []
        next_q_values = []
        
        for state, next_state in zip(states, next_states):
            # Current state Q-values
            char_indices, guessed, word_length_bits, vowel_ratio, lives = prepare_input(state, self.device)
            q_vals = self.policy_net(char_indices, guessed, word_length_bits, vowel_ratio, lives)
            current_q_values.append(q_vals)
            
            # Next state Q-values from target network
            if next_state:
                char_indices, guessed, word_length_bits, vowel_ratio, lives = prepare_input(next_state, self.device)
                with torch.no_grad():
                    next_q = self.target_net(char_indices, guessed, word_length_bits, vowel_ratio, lives)
                next_q_values.append(next_q)
            else:
                next_q_values.append(torch.zeros_like(q_vals))
        
        current_q_values = torch.cat(current_q_values)
        next_q_values = torch.cat(next_q_values)
        
        # Get Q-values for taken actions
        action_indices = torch.tensor([ord(a) - ord('a') for a in actions], device=self.device)
        current_q = current_q_values.gather(1, action_indices.unsqueeze(1))
        
        # Compute target Q-values
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        next_q_max = next_q_values.max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q_max
        
        # Compute loss and update
        loss = F.smooth_l1_loss(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()

def create_curriculum_dataset(words, num_missing_letters):
    """Create dataset with specified number of missing letters"""
    states = []
    for word in words:
        # Choose random positions to mask
        word_length = len(word)
        num_to_mask = min(num_missing_letters, word_length - 1)  # Leave at least one letter
        mask_positions = random.sample(range(word_length), num_to_mask)
        
        # Create initial state
        current_state = list(word)
        for pos in mask_positions:
            current_state[pos] = '_'
        
        state = {
            'current_state': ''.join(current_state),
            'guessed_letters': [],
            'original_word': word,
            'vowel_ratio': sum(1 for c in word if c in 'aeiou') / word_length,
            'remaining_lives': 6
        }
        states.append(state)
    
    return states

def calculate_reward(correct_guess, remaining_lives, revealed_ratio, done=False, won=False, repeated_guess=False):
    """Enhanced reward calculation"""
    if repeated_guess:
        return -10.0  # Heavy penalty for repeated guesses
    
    if done:
        if won:
            return 20.0  # Big reward for winning
        else:
            return -15.0  # Big penalty for losing
    
    # Regular game rewards
    if correct_guess:
        return 1.0 + revealed_ratio
    else:
        return -1.0 * (1.2 ** (6 - remaining_lives))

def load_saved_data(data_path):
    """Load saved dataset from pickle file"""
    logging.info(f"Loading saved dataset from {data_path}")
    try:
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            train_states = data.get('train_states')
            val_states = data.get('val_states')
            val_words = data.get('val_words')
            
            if not all([train_states, val_states, val_words]):
                raise ValueError("Missing data in saved dataset")
                
            logging.info(f"Loaded {len(train_states)} training states")
            logging.info(f"Loaded {len(val_states)} validation states")
            logging.info(f"Loaded {len(val_words)} validation words")
            
            return train_states, val_states, val_words
    except Exception as e:
        logging.error(f"Failed to load dataset: {str(e)}")
        raise

def load_and_split_words(word_file='words_250000_train.txt', train_ratio=0.8):
    """Load words from dictionary and split into train/val sets"""
    logging.info(f"Loading words from {word_file}")
    with open(word_file, 'r') as f:
        words = [word.strip().lower() for word in f.readlines()]
    
    # Filter words (optional)
    words = [w for w in words if len(w) >= 4 and len(w) <= 15]  # reasonable lengths
    
    # Shuffle and split
    random.shuffle(words)
    split_idx = int(len(words) * train_ratio)
    train_words = words[:split_idx]
    val_words = words[split_idx:]
    
    logging.info(f"Loaded {len(words)} words")
    logging.info(f"Training words: {len(train_words)}")
    logging.info(f"Validation words: {len(val_words)}")
    
    return train_words, val_words

def play_complete_game(word, agent, epsilon):
    """Play a complete game of hangman for a given word"""
    state = {
        'current_state': '_' * len(word),
        'guessed_letters': [],
        'original_word': word,
        'vowel_ratio': sum(1 for c in word if c in 'aeiou') / len(word),
        'remaining_lives': 6
    }
    
    experiences = []
    game_over = False
    
    while not game_over and state['remaining_lives'] > 0 and '_' in state['current_state']:
        # Get action from agent
        action = agent.get_action(state, epsilon)
        
        # Handle repeated guesses
        if action in state['guessed_letters']:
            reward = calculate_reward(False, state['remaining_lives'], 0, done=True, won=False, repeated_guess=True)
            next_state = None
            done = True
            game_over = True
            experiences.append((state, action, reward, next_state, done))
            continue
            
        # Apply action and get reward
        correct_guess = action in word
        revealed_ratio = 1 - state['current_state'].count('_') / len(word)
        
        # Create next state
        next_state = state.copy()
        next_state['guessed_letters'] = state['guessed_letters'] + [action]
        
        if correct_guess:
            current = list(state['current_state'])
            for i, letter in enumerate(word):
                if letter == action:
                    current[i] = letter
            next_state['current_state'] = ''.join(current)
        else:
            next_state['remaining_lives'] -= 1
        
        done = (next_state['remaining_lives'] == 0 or '_' not in next_state['current_state'])
        won = '_' not in next_state['current_state']
        
        # Calculate reward with game completion bonuses
        reward = calculate_reward(
            correct_guess, 
            state['remaining_lives'], 
            revealed_ratio,
            done=done,
            won=won
        )
        
        if done:
            game_over = True
            
        experiences.append((state, action, reward, None if done else next_state, done))
        state = next_state
    
    # Check if game was won
    won = state is not None and '_' not in state['current_state']
    return experiences, won

def test_model(agent, test_words, verbose=False):
    """Test model on a set of words and return statistics"""
    wins = 0
    total_games = len(test_words)
    game_logs = []
    
    for word in test_words:
        experiences, won = play_complete_game(word, agent, epsilon=0)
        wins += int(won)
        
        if verbose and len(game_logs) < 5:
            # Log the game play
            game_log = [f"\nTesting word: {word}"]
            for exp in experiences:
                state, action, reward, next_state, done = exp
                correct = action in word
                game_log.append(
                    f"Guessed: {action} ({'✓' if correct else '✗'}) "
                    f"State: {state['current_state']} "
                    f"Lives: {state['remaining_lives']}"
                )
            game_log.append(f"Result: {'Success' if won else 'Failure'}")
            game_logs.append('\n'.join(game_log))
    
    return wins/total_games, game_logs

def create_training_states(words):
    """Create all possible states for each word with different missing letters"""
    all_states = []
    
    for word in words:
        word_len = len(word)
        vowel_ratio = sum(1 for c in word if c in 'aeiou') / word_len
        
        # Create states with different numbers of missing letters
        for num_missing in range(1, min(word_len, 11)):  # Up to 10 missing letters
            # Generate all possible combinations of missing letters
            for _ in range(min(5, word_len)):  # Limit combinations per word
                mask_positions = random.sample(range(word_len), num_missing)
                current_state = list(word)
                for pos in mask_positions:
                    current_state[pos] = '_'
                
                state = {
                    'current_state': ''.join(current_state),
                    'guessed_letters': [],
                    'original_word': word,
                    'vowel_ratio': vowel_ratio,
                    'remaining_lives': 6,
                    'num_missing': num_missing
                }
                all_states.append(state)
    
    return all_states

def sort_states(states):
    """Sort states by number of missing letters and then by length"""
    return sorted(states, key=lambda x: (x['num_missing'], len(x['current_state'])))

def count_missing_letters(state):
    """Count number of missing letters in a state"""
    return state['current_state'].count('_')

def prepare_loaded_states(states):
    """Add missing letter count to loaded states and sort"""
    for state in states:
        state['num_missing'] = count_missing_letters(state)
    return states

def inspect_input_dimensions(state, device):
    """Print dimensions of all input tensors for a single state"""
    char_indices, guessed, word_length_bits, vowel_ratio, lives = prepare_input(state, device)
    
    logging.info("\nInput tensor dimensions:")
    logging.info(f"char_indices: {char_indices.shape}, {char_indices.dtype}")
    logging.info(f"Sample char_indices: {char_indices[:10]}")  # Show first 10 indices
    logging.info(f"guessed: {guessed.shape}, {guessed.dtype}")
    logging.info(f"word_length_bits: {word_length_bits.shape}, {word_length_bits.dtype}")
    logging.info(f"vowel_ratio: {vowel_ratio.shape}, {vowel_ratio.dtype}")
    logging.info(f"lives: {lives.shape}, {lives.dtype}")
    
    # Test network forward pass
    model = HangmanDQN().to(device)
    with torch.no_grad():
        # Add batch dimension for network
        char_indices = char_indices.unsqueeze(0)  # [1, seq_len]
        guessed = guessed.unsqueeze(0)  # [1, 26]
        output = model(char_indices, guessed, word_length_bits, vowel_ratio, lives)
        logging.info(f"\nNetwork output shape: {output.shape}")
        
    return char_indices.shape, guessed.shape, word_length_bits.shape, vowel_ratio.shape, lives.shape

def main():
    log_file = setup_logging()
    logging.info("Starting reinforcement learning training...")
    
    # Configuration
    QUICK_TEST = True
    DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    SAVED_DATA_PATH = 'hangman_data/hangman_states_nv3_20250201_122702.pkl'
    
    try:
        # Load saved dataset
        logging.info("Loading dataset...")
        train_states, val_states, val_words = load_saved_data(SAVED_DATA_PATH)
        
        # Inspect dimensions of a sample state
        logging.info("\nInspecting input dimensions for a sample state...")
        sample_state = train_states[0]
        logging.info(f"Sample state: {sample_state}")
        input_shapes = inspect_input_dimensions(sample_state, DEVICE)
        logging.info(f"Input shapes: {input_shapes}")
        
        if QUICK_TEST:
            train_states = random.sample(train_states, len(train_states) // 10)
            val_states = random.sample(val_states, len(val_states) // 10)
            val_words = random.sample(val_words, len(val_words) // 10)
        
        # Prepare states for curriculum learning
        logging.info("Preparing states for curriculum learning...")
        train_states = prepare_loaded_states(train_states)
        logging.info(f"Prepared {len(train_states)} training states")
        
        EPOCHS = 10 if QUICK_TEST else 50
        BATCH_SIZE = 32 if QUICK_TEST else 64
        INITIAL_EPSILON = 0.5
        FINAL_EPSILON = 0.1
        
        # Initialize agent
        logging.info("\nInitializing agent...")
        agent = HangmanAgent(device=DEVICE)
        
        # Training loop
        best_val_win_rate = 0
        patience = 5
        patience_counter = 0
        
        for epoch in range(EPOCHS):
            logging.info(f"\nEpoch {epoch + 1}/{EPOCHS}")
            epsilon = INITIAL_EPSILON - (INITIAL_EPSILON - FINAL_EPSILON) * (epoch / EPOCHS)
            
            # Sort states by missing letters
            sorted_states = sort_states(train_states)
            current_missing = 1
            current_batch = []
            total_loss = 0
            num_batches = 0
            
            # Process states in curriculum order
            for state in tqdm(sorted_states, desc=f"Epoch {epoch + 1}"):
                if state['num_missing'] > current_missing:
                    # Process accumulated batch
                    if current_batch:
                        random.shuffle(current_batch)
                        for i in range(0, len(current_batch), BATCH_SIZE):
                            batch_states = current_batch[i:i + BATCH_SIZE]
                            if len(batch_states) >= BATCH_SIZE:
                                # Get actions and next states
                                experiences = []
                                for s in batch_states:
                                    action = agent.get_action(s, epsilon)
                                    word = s['original_word']
                                    
                                    # Create next state
                                    next_s = s.copy()
                                    next_s['guessed_letters'] = s['guessed_letters'] + [action]
                                    
                                    if action in word:
                                        current = list(s['current_state'])
                                        for i, letter in enumerate(word):
                                            if letter == action:
                                                current[i] = letter
                                        next_s['current_state'] = ''.join(current)
                                        reward = 1.0
                                    else:
                                        next_s['remaining_lives'] -= 1
                                        reward = -1.0
                                    
                                    done = next_s['remaining_lives'] == 0
                                    experiences.append((s, action, reward, None if done else next_s, done))
                                
                                # Add to replay buffer
                                for exp in experiences:
                                    agent.replay_buffer.append(exp)
                                
                                # Train on random batch from replay buffer
                                if len(agent.replay_buffer) >= BATCH_SIZE:
                                    batch = random.sample(agent.replay_buffer, BATCH_SIZE)
                                    loss = agent.train_step(batch)
                                    total_loss += loss
                                    num_batches += 1
                    
                    current_missing = state['num_missing']
                    current_batch = []
                
                current_batch.append(state)
            
            # Log training metrics
            avg_loss = total_loss / max(1, num_batches)
            logging.info(f"Average loss: {avg_loss:.4f}")
            logging.info(f"Replay buffer size: {len(agent.replay_buffer)}")
            
            # Validation
            if (epoch + 1) % 1 == 0:
                val_wins = 0
                val_games = 0
                logging.info("\nValidation:")
                
                # Test on complete games
                test_words = random.sample(val_words, min(100, len(val_words)))
                val_win_rate, game_logs = test_model(agent, test_words, verbose=True)
                
                # Log example games
                for log in game_logs:
                    logging.info(log)
                
                logging.info(f"Validation win rate: {val_win_rate:.2%}")
                
                # Early stopping check
                if val_win_rate > best_val_win_rate:
                    best_val_win_rate = val_win_rate
                    patience_counter = 0
                    torch.save({
                        'policy_net': agent.policy_net.state_dict(),
                        'target_net': agent.target_net.state_dict(),
                        'val_win_rate': val_win_rate,
                        'epoch': epoch
                    }, f'hangman_data/dqn_model_best.pt')
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logging.info(f"\nEarly stopping! Best validation win rate: {best_val_win_rate:.2%}")
                        break
        
        logging.info("\nTraining complete!")
        
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise
    finally:
        logging.info(f"Log file saved to: {log_file}")

if __name__ == "__main__":
    main() 