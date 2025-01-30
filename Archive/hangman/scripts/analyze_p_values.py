import sys
import os
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Add project root to Python path
try:
    # When running as a script
    project_root = Path(__file__).parent.parent.parent
except NameError:
    # When running in notebook
    project_root = Path.cwd().parent

# Ensure project_root exists and contains hangman module
if not (project_root / 'hangman').exists():
    raise RuntimeError(
        f"Could not find hangman module in {project_root}. "
        "Please run this script from the correct directory."
    )

# Add to Python path
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Now try imports
try:
    from hangman.data.word_processor import WordProcessor
    from hangman.data.simulator import HangmanSimulator
except ModuleNotFoundError as e:
    print(f"Error: {e}")
    print(f"Python path: {sys.path}")
    print(f"Project root: {project_root}")
    print(f"Contents of {project_root}:")
    for p in project_root.iterdir():
        print(f"  {p}")
    raise

def analyze_dataset_stats(stats):
    """Analyze and display dataset statistics"""
    print("\nDataset Statistics:")
    print(f"Total Games: {stats['total_games']}")
    print(f"Win Rate: {stats['wins']/stats['total_games']*100:.2f}%")
    print(f"Average Guesses: {stats['total_guesses']/stats['total_games']:.2f}")
    
    # Analyze by word length
    lengths = sorted(stats['by_length'].keys())
    length_stats = []
    for length in lengths:
        data = stats['by_length'][length]
        length_stats.append({
            'length': length,
            'games': data['total_games'],
            'win_rate': data['wins']/data['total_games']*100,
            'avg_guesses': data['total_guesses']/data['total_games']
        })
    
    df = pd.DataFrame(length_stats)
    print("\nStats by word length:")
    print(df.to_string())
    return df

def plot_comparative_stats(all_stats, p_values, save_dir):
    """Create comparative plots for different p values"""
    plt.style.use('seaborn')
    sns.set_palette('husl')
    
    # 1. Win Rates
    plt.figure(figsize=(10, 6))
    win_rates = [stats['wins']/stats['total_games']*100 for stats in all_stats.values()]
    sns.barplot(x=[f"p={p}" for p in p_values], y=win_rates)
    plt.title('Win Rates by p Value')
    plt.ylabel('Win Rate (%)')
    plt.savefig(f'{save_dir}/win_rates.png')
    plt.close()
    
    # 2. Average Guesses
    plt.figure(figsize=(10, 6))
    avg_guesses = [stats['total_guesses']/stats['total_games'] for stats in all_stats.values()]
    sns.barplot(x=[f"p={p}" for p in p_values], y=avg_guesses)
    plt.title('Average Guesses by p Value')
    plt.ylabel('Average Guesses')
    plt.savefig(f'{save_dir}/avg_guesses.png')
    plt.close()
    
    # 3. Game Length Distributions
    plt.figure(figsize=(12, 6))
    for p, stats in all_stats.items():
        sns.kdeplot(stats['game_lengths'], label=f"p={p}")
    plt.title('Distribution of Game Lengths')
    plt.xlabel('Number of Guesses')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(f'{save_dir}/game_lengths_dist.png')
    plt.close()
    
    # 4. Success Rate by Word Length
    plt.figure(figsize=(15, 6))
    for p, stats in all_stats.items():
        lengths = sorted(stats['by_length'].keys())
        success_rates = [
            stats['by_length'][l]['wins'] / stats['by_length'][l]['total_games'] * 100 
            for l in lengths
        ]
        plt.plot(lengths, success_rates, marker='o', label=f"p={p}")
    plt.title('Success Rate by Word Length')
    plt.xlabel('Word Length')
    plt.ylabel('Success Rate (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{save_dir}/success_rate_by_length.png')
    plt.close()

def create_summary_df(all_stats, p_values):
    """Create summary DataFrame of all statistics"""
    summary_data = []
    for p in p_values:
        stats = all_stats[p]
        summary_data.append({
            'p_value': p,
            'win_rate': stats['wins']/stats['total_games']*100,
            'avg_guesses': stats['total_guesses']/stats['total_games'],
            'total_games': stats['total_games'],
            'total_states': sum(len(stats['by_length'][l]['guesses_list']) 
                              for l in stats['by_length']),
            'avg_states_per_word': sum(len(stats['by_length'][l]['guesses_list']) 
                                     for l in stats['by_length'])/stats['total_games']
        })
    
    df = pd.DataFrame(summary_data)
    print("\nSummary Statistics:")
    print(df.to_string(float_format=lambda x: '{:.2f}'.format(x)))
    return df

def main():
    # Configuration
    p_values = [0.3, 0.5, 0.7, 0.9]
    word_file = os.path.join(project_root, "words_250000_train.txt")
    base_save_dir = os.path.join(project_root, "data/p_analysis")
    os.makedirs(base_save_dir, exist_ok=True)
    
    print(f"Using word file: {word_file}")
    print(f"Saving results to: {base_save_dir}")
    
    # Load words
    with open(word_file, 'r') as f:
        words = [word.strip().lower() for word in f.readlines()]
    
    # Optional: Use sample for quick testing
    sample_size = None  # Set to a number (e.g., 1000) for quick testing
    if sample_size:
        np.random.seed(42)
        words = np.random.choice(words, sample_size, replace=False)
        print(f"Using {sample_size} words for quick analysis")
    
    # Generate and analyze datasets
    all_stats = {}
    for p in p_values:
        print(f"\nGenerating dataset with p = {p}")
        save_dir = f"{base_save_dir}/p_{int(p*100)}"
        os.makedirs(save_dir, exist_ok=True)
        
        word_processor = WordProcessor(words)
        simulator = HangmanSimulator(word_processor, optimal_prob=p)
        
        simulator.generate_training_data(words, f"{save_dir}/states.pt")
        simulator.plot_statistics(save_dir)
        
        all_stats[p] = simulator.stats
        analyze_dataset_stats(simulator.stats)
    
    # Create comparative plots
    plot_comparative_stats(all_stats, p_values, base_save_dir)
    
    # Generate summary
    summary_df = create_summary_df(all_stats, p_values)
    summary_df.to_csv(f"{base_save_dir}/summary_stats.csv", index=False)

if __name__ == "__main__":
    main() 