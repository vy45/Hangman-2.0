import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class HangmanEnvironment:
    def __init__(self, word):
        self.word = word.lower()
        self.word_length = len(word)
        self.max_word_length = 30
        self.reset()
    
    def reset(self):
        self.guessed_letters = set()
        self.num_guesses = 0
        self.current_state = ['_'] * self.word_length
        return self._get_state_representation()
    
    def _get_state_representation(self):
        # Create binary vector for guessed letters
        alphabet_state = np.zeros(26)
        for letter in self.guessed_letters:
            idx = ord(letter) - ord('a')
            alphabet_state[idx] = 1
        
        # Create word state matrix (max_length × 27)
        word_state = np.zeros((self.max_word_length, 27))
        for i in range(self.word_length):
            if self.current_state[i] == '_':
                word_state[i, 26] = 1  # mask token
            else:
                word_state[i, ord(self.current_state[i]) - ord('a')] = 1
                
        # Pad remaining positions
        for i in range(self.word_length, self.max_word_length):
            word_state[i, 26] = 1
        
        return {
            'alphabet_state': alphabet_state,
            'word_state': word_state,
            'num_guesses': self.num_guesses
        }
    
    def step(self, action):
        letter = chr(action + ord('a'))
        
        if letter in self.guessed_letters:
            return self._get_state_representation(), -5.0, True  # Heavy penalty for repeated guess
        
        self.guessed_letters.add(letter)
        self.num_guesses += 1
        
        # Check if letter is in word
        found = False
        num_occurrences = 0
        for i, char in enumerate(self.word):
            if char == letter:
                self.current_state[i] = letter
                found = True
                num_occurrences += 1
        
        # Calculate reward
        if found:
            reward = num_occurrences  # Reward for each correct letter
            if '_' not in self.current_state:
                reward += 10.0  # Bonus for completing word
                done = True
            else:
                done = False
        else:
            reward = -np.log(self.num_guesses + 1)  # Logarithmic penalty
            done = False
        
        return self._get_state_representation(), reward, done

class HangmanDQN(nn.Module):
    def __init__(self, max_word_length=30):
        super(HangmanDQN, self).__init__()
        
        self.conv = nn.Conv1d(
            in_channels=27,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            batch_first=True
        )
        
        self.combine_features = nn.Sequential(
            nn.Linear(128 + 26 + 1, 64),  # LSTM output + alphabet + num_guesses
            nn.ReLU(),
            nn.Linear(64, 26)
        )
    
    def forward(self, state):
        # Process word state with CNN
        word_state = state['word_state'].transpose(1, 2)  # Prepare for Conv1D
        conv_out = self.conv(word_state)
        conv_out = conv_out.transpose(1, 2)  # Prepare for LSTM
        
        # Process with LSTM
        lstm_out, _ = self.lstm(conv_out)
        lstm_features = lstm_out[:, -1, :]  # Take final output
        
        # Combine with game state
        num_guesses = torch.tensor([state['num_guesses']], dtype=torch.float32).unsqueeze(0)
        combined = torch.cat([
            lstm_features,
            state['alphabet_state'],
            num_guesses
        ], dim=1)
        
        # Get Q-values
        q_values = self.combine_features(combined)
        
        # Mask already guessed letters
        mask = (state['alphabet_state'] == 1)
        q_values[mask] = float('-inf')
        
        return q_values

class MetricsTracker:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.metrics = {
            'words_solved': deque(maxlen=window_size),
            'guesses_to_solve': deque(maxlen=window_size),
            'success_within_6': deque(maxlen=window_size),
            'episode_rewards': deque(maxlen=window_size)
        }
    
    def update(self, solved, num_guesses, episode_reward):
        self.metrics['words_solved'].append(int(solved))
        self.metrics['guesses_to_solve'].append(num_guesses)
        self.metrics['success_within_6'].append(1 if solved and num_guesses <= 6 else 0)
        self.metrics['episode_rewards'].append(episode_reward)
    
    def get_stats(self):
        return {
            'solve_rate': np.mean(self.metrics['words_solved']),
            'avg_guesses': np.mean(self.metrics['guesses_to_solve']),
            'success_rate_6': np.mean(self.metrics['success_within_6']),
            'mean_reward': np.mean(self.metrics['episode_rewards'])
        }
    
    def plot_metrics(self):
        stats = self.get_stats()
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot metrics
        metrics_list = list(self.metrics.items())
        titles = ['Words Solved Rate', 'Average Guesses', 'Success Rate (≤6)', 'Mean Reward']
        
        for idx, ((metric_name, metric_data), title) in enumerate(zip(metrics_list, titles)):
            ax = axes[idx//2, idx%2]
            ax.plot(list(metric_data))
            ax.set_title(f'{title}: {stats[metric_name.replace("s", "_rate") if "solved" in metric_name else metric_name]:.3f}')
            ax.grid(True)
        
        plt.tight_layout()
        plt.show()

def train_hangman_agent(word_list, num_episodes=10000, batch_size=32, memory_size=10000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = HangmanDQN().to(device)
    target_net = HangmanDQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    
    optimizer = optim.Adam(policy_net.parameters())
    memory = deque(maxlen=memory_size)
    metrics = MetricsTracker()
    
    epsilon = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995
    
    for episode in tqdm(range(num_episodes)):
        word = random.choice(word_list)
        env = HangmanEnvironment(word)
        state = env.reset()
        episode_reward = 0
        
        while True:
            # Select action
            if random.random() < epsilon:
                action = random.randint(0, 25)
                while state['alphabet_state'][action] == 1:  # Avoid already guessed
                    action = random.randint(0, 25)
            else:
                with torch.no_grad():
                    q_values = policy_net(state)
                    action = q_values.argmax().item()
            
            # Take step
            next_state, reward, done = env.step(action)
            episode_reward += reward
            
            # Store experience
            memory.append(Experience(state, action, reward, next_state, done))
            state = next_state
            
            # Train on batch
            if len(memory) >= batch_size:
                experiences = random.sample(memory, batch_size)
                batch = Experience(*zip(*experiences))
                
                state_batch = {k: torch.stack([torch.tensor(s[k]) for s in batch.state]) 
                             for k in batch.state[0].keys()}
                next_state_batch = {k: torch.stack([torch.tensor(s[k]) for s in batch.next_state]) 
                                  for k in batch.next_state[0].keys()}
                
                action_batch = torch.tensor(batch.action).unsqueeze(1)
                reward_batch = torch.tensor(batch.reward)
                done_batch = torch.tensor(batch.done)
                
                # Compute current Q values
                current_q = policy_net(state_batch).gather(1, action_batch)
                
                # Compute next Q values
                next_q = target_net(next_state_batch).max(1)[0].detach()
                target_q = reward_batch + (1 - done_batch) * 0.99 * next_q
                
                # Compute loss and update
                loss = nn.MSELoss()(current_q.squeeze(), target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if done:
                break
        
        # Update metrics
        metrics.update(
            solved='_' not in env.current_state,
            num_guesses=env.num_guesses,
            episode_reward=episode_reward
        )
        
        # Update target network periodically
        if episode % 100 == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # Plot metrics periodically
        if episode % 1000 == 0:
            metrics.plot_metrics()
    
    return policy_net, metrics

# Usage example:
"""
word_list = ["example", "words", "here"]  # Replace with actual word list
model, metrics = train_hangman_agent(word_list)
metrics.plot_metrics()
"""
