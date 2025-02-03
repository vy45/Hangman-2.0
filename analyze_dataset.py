import pickle
import random
import csv
from collections import defaultdict
import logging
from pathlib import Path

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def analyze_dataset(data_path, output_path):
    logging.info(f"Loading dataset from {data_path}")
    
    # Load the dataset
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
        train_states = data.get('train_states', [])
    
    # Group states by word length
    length_groups = defaultdict(list)
    for state in train_states:
        word_len = len(state['original_word'])
        length_groups[word_len].append(state)
    
    # Prepare data for CSV
    rows = []
    headers = [
        'word_length',
        'original_word',
        'current_state',
        'guessed_letters',
        'remaining_lives',
        'vowel_ratio',
        'target_distribution'
    ]
    
    logging.info("Sampling states for each word length...")
    
    # Sample 2 states from each length group
    for length, states in sorted(length_groups.items()):
        samples = random.sample(states, min(2, len(states)))
        for state in samples:
            row = {
                'word_length': length,
                'original_word': state['original_word'],
                'current_state': state['current_state'],
                'guessed_letters': ','.join(state['guessed_letters']),
                'remaining_lives': state['remaining_lives'],
                'vowel_ratio': state['vowel_ratio'],
                'target_distribution': str(state.get('target_distribution', {}))
            }
            rows.append(row)
    
    # Write to CSV
    logging.info(f"Writing samples to {output_path}")
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    
    # Print summary
    logging.info("\nDataset Summary:")
    for length in sorted(length_groups.keys()):
        logging.info(f"Words of length {length}: {len(length_groups[length])} states")

if __name__ == "__main__":
    setup_logging()
    data_path = 'hangman_data/hangman_states_nv3_20250201_122702.pkl'
    output_path = 'dataset_samples.csv'
    
    try:
        analyze_dataset(data_path, output_path)
        logging.info("Analysis complete! Check dataset_samples.csv for results")
    except Exception as e:
        logging.error(f"Analysis failed: {str(e)}")
        raise 