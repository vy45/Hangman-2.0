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
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_file

class ActorCriticHead(nn.Module):
    def __init__(self, lstm_hidden_dim=26):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(lstm_hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 26),  # Output logits for each letter
        )
        
        self.critic = nn.Sequential(
            nn.Linear(lstm_hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Value estimate
        )
    
    def forward(self, lstm_features):
        return self.actor(lstm_features), self.critic(lstm_features)

class HangmanRLAgent:
    def __init__(self, pretrained_path, device='cpu', lr=1e-3, gamma=0.99):
        self.device = device
        self.gamma = gamma
        
        logging.info("Initializing HangmanRLAgent...")
        
        # Load pretrained LSTM with higher learning potential
        logging.info(f"Loading pretrained model from {pretrained_path}")
        self.base_model = MaryLSTMModel().to(device)
        logging.info("Base model created and moved to device")
        
        try:
            logging.info("Loading checkpoint...")
            checkpoint = torch.load(pretrained_path, map_location=device, weights_only=True)
            logging.info("Checkpoint loaded, applying state dict...")
            self.base_model.load_state_dict(checkpoint['model_state_dict'])
            logging.info("Model weights loaded successfully")
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise
        
        # Add actor-critic heads
        logging.info("Initializing actor-critic heads...")
        self.ac_head = ActorCriticHead(lstm_hidden_dim=26).to(device)
        
        # Optimizer with different learning rates for base and AC head
        logging.info("Setting up optimizer...")
        self.optimizer = torch.optim.Adam([
            {'params': self.base_model.parameters(), 'lr': lr},
            {'params': self.ac_head.parameters(), 'lr': lr * 0.1}  # Lower lr for AC head
        ])
        
        # Experience replay buffer
        self.replay_buffer = deque(maxlen=100000)  # Increased buffer size
        logging.info("Agent initialization complete")
        
    def get_action(self, state, epsilon):
        """Select action using epsilon-greedy policy"""
        with torch.no_grad():
            # Get LSTM features
            char_indices, guessed, word_length, vowel_ratio, lives = prepare_input(state, self.device)
            
            # Get probabilities from base model
            probs = self.base_model(
                char_indices.unsqueeze(0), 
                guessed.unsqueeze(0),
                word_length,
                vowel_ratio,
                lives
            )
            
            # Use probabilities as features for actor-critic
            action_logits, value = self.ac_head(probs)
            
            # Mask already guessed letters
            for letter in state['guessed_letters']:
                action_logits[0, ord(letter) - ord('a')] = float('-inf')
            
            if random.random() < epsilon:
                # Random action from unguessed letters
                valid_actions = [i for i in range(26) 
                               if chr(i + ord('a')) not in state['guessed_letters']]
                action_idx = random.choice(valid_actions)
            else:
                # Greedy action
                action_idx = action_logits.argmax().item()
            
            return chr(action_idx + ord('a'))
    
    def train_step(self, batch):
        """Train on a batch of experiences"""
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Get current state values and action logits
        current_features = []
        next_features = []
        
        for state, next_state in zip(states, next_states):
            # Process current state
            char_indices, guessed, word_length, vowel_ratio, lives = prepare_input(state, self.device)
            current_features.append(
                self.base_model(
                    char_indices.unsqueeze(0),
                    guessed.unsqueeze(0),
                    word_length,
                    vowel_ratio,
                    lives
                )
            )
            
            # Process next state
            if next_state:
                char_indices, guessed, word_length, vowel_ratio, lives = prepare_input(next_state, self.device)
                next_features.append(
                    self.base_model(
                        char_indices.unsqueeze(0),
                        guessed.unsqueeze(0),
                        word_length,
                        vowel_ratio,
                        lives
                    )
                )
            else:
                next_features.append(torch.zeros_like(current_features[-1]))
        
        current_features = torch.cat(current_features)
        next_features = torch.cat(next_features)
        
        # Get values and action logits
        action_logits, state_values = self.ac_head(current_features)
        _, next_values = self.ac_head(next_features)
        
        # Compute advantages
        with torch.no_grad():
            advantages = rewards + self.gamma * next_values * (1 - dones) - state_values
        
        # Compute action log probs
        action_indices = torch.tensor([ord(a) - ord('a') for a in actions]).to(self.device)
        action_log_probs = F.log_softmax(action_logits, dim=-1)
        selected_action_log_probs = action_log_probs.gather(1, action_indices.unsqueeze(-1))
        
        # Compute losses
        actor_loss = -(advantages.detach() * selected_action_log_probs).mean()
        critic_loss = advantages.pow(2).mean()
        
        # Update model
        total_loss = actor_loss + 0.5 * critic_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()

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

def prepare_input(state, device):
    """Convert state to model input tensors"""
    # logging.info("\nPreparing input state:")
    # logging.info(f"State keys: {state.keys()}")
    # logging.info(f"Current state: {state['current_state']}")
    # logging.info(f"Guessed letters: {state['guessed_letters']}")
    
    try:
        # Convert current state to char indices
        char_indices = torch.zeros(32, dtype=torch.long, device=device)
        for i, char in enumerate(state['current_state']):
            if char == '_':
                char_indices[i] = 27
            else:
                char_indices[i] = ord(char.lower()) - ord('a') + 1
        # logging.info(f"Char indices shape: {char_indices.shape}")
                
        # Create guessed letters tensor
        guessed = torch.zeros(26, device=device)
        for letter in state['guessed_letters']:
            guessed[ord(letter) - ord('a')] = 1
        # logging.info(f"Guessed tensor shape: {guessed.shape}")
        
        # Word length
        word_length = torch.tensor([len(state['current_state'])], dtype=torch.float32, device=device)
        # logging.info(f"Word length: {word_length.item()}")
        
        # Vowel ratio
        vowel_ratio = torch.tensor([state['vowel_ratio']], dtype=torch.float32, device=device)
        # logging.info(f"Vowel ratio: {vowel_ratio.item()}")
        
        # Remaining lives
        remaining_lives = torch.tensor([state['remaining_lives']], dtype=torch.float32, device=device)
        # logging.info(f"Remaining lives: {remaining_lives.item()}")
        
        # Create target distribution if not present
        if 'target_distribution' not in state:
            # logging.info("Creating target distribution")
            target_dist = {l: 0.0 for l in string.ascii_lowercase}
            remaining_letters = set(state['original_word']) - set(state['guessed_letters'])
            for letter in remaining_letters:
                target_dist[letter] = 1.0
            # Normalize
            total = sum(target_dist.values())
            if total > 0:
                target_dist = {k: v/total for k, v in target_dist.items()}
            state['target_distribution'] = target_dist
        
        # logging.info("Input preparation complete")
        return char_indices, guessed, word_length, vowel_ratio, remaining_lives
        
    except Exception as e:
        logging.error(f"Error in prepare_input: {str(e)}")
        logging.error(f"State: {state}")
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

def main():
    log_file = setup_logging()
    logging.info("Starting reinforcement learning training...")
    
    # Configuration
    QUICK_TEST = True
    DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    PRETRAINED_PATH = 'hangman_data/mary_model_nv3_epochfinal_20250201_173921.pt'
    
    # Load and split words
    train_words, val_words = load_and_split_words()
    
    if QUICK_TEST:
        train_words = random.sample(train_words, len(train_words) // 10)
        val_words = random.sample(val_words, len(val_words) // 10)
    
    EPOCHS = 10 if QUICK_TEST else 50
    BATCH_SIZE = 32 if QUICK_TEST else 64
    INITIAL_EPSILON = 0.5
    FINAL_EPSILON = 0.1
    
    try:
        # Initialize agent
        logging.info("\nInitializing RL agent...")
        agent = HangmanRLAgent(PRETRAINED_PATH, device=DEVICE)
        
        # Early stopping setup
        best_val_win_rate = 0
        patience = 5
        patience_counter = 0
        
        # Training loop
        for epoch in range(EPOCHS):
            logging.info(f"\nEpoch {epoch + 1}/{EPOCHS}")
            epsilon = INITIAL_EPSILON - (INITIAL_EPSILON - FINAL_EPSILON) * (epoch / EPOCHS)
            
            # Training
            total_loss = 0
            num_batches = 0
            wins = 0
            games = 0
            
            # Process all training words
            logging.info(f"Processing {len(train_words)} words...")
            all_experiences = []
            
            # Create 5 games for each word simultaneously
            for word in tqdm(train_words, desc=f"Epoch {epoch + 1} - Collecting experiences"):
                # Play 5 games for each word
                word_experiences = []
                for _ in range(5):
                    exp, won = play_complete_game(word, agent, epsilon)
                    word_experiences.extend(exp)
                    games += 1
                    wins += int(won)
                
                # Add experiences to replay buffer
                for exp in word_experiences:
                    agent.replay_buffer.append(exp)
                
                # Train when buffer has enough samples
                if len(agent.replay_buffer) >= BATCH_SIZE:
                    # Sample multiple batches from replay buffer
                    for _ in range(5):  # Train more frequently on experiences
                        batch = random.sample(agent.replay_buffer, BATCH_SIZE)
                        loss = agent.train_step(batch)
                        total_loss += loss
                        num_batches += 1
                        
                        if QUICK_TEST and num_batches % 10 == 0:
                            logging.info(f"Batch {num_batches}: loss = {loss:.4f}")
            
            avg_loss = total_loss / max(1, num_batches)
            win_rate = wins / max(1, games)
            logging.info(f"Average loss: {avg_loss:.4f}")
            logging.info(f"Win rate: {win_rate:.2%}")
            logging.info(f"Replay buffer size: {len(agent.replay_buffer)}")
            
            # Validation
            val_wins = 0
            val_games = 0
            logging.info("\nValidation:")
            
            for word in random.sample(val_words, min(100, len(val_words))):
                _, won = play_complete_game(word, agent, 0.0)  # No exploration
                val_games += 1
                val_wins += int(won)
            
            val_win_rate = val_wins / max(1, val_games)
            logging.info(f"Validation win rate: {val_win_rate:.2%}")
            
            # Early stopping check
            if val_win_rate > best_val_win_rate:
                best_val_win_rate = val_win_rate
                patience_counter = 0
                # Save best model
                torch.save({
                    'base_model_state_dict': agent.base_model.state_dict(),
                    'ac_head_state_dict': agent.ac_head.state_dict(),
                    'val_win_rate': val_win_rate,
                    'epoch': epoch
                }, f'hangman_data/rl_model_best.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info(f"\nEarly stopping triggered! Best validation win rate: {best_val_win_rate:.2%}")
                    break
        
        logging.info("\nTraining complete!")
        
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise
    finally:
        logging.info(f"Log file saved to: {log_file}")

if __name__ == "__main__":
    main() 