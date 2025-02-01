# Mary Hangman NV4

A deep learning model for playing Hangman using a bidirectional LSTM architecture.

## Setup

1. Make sure you have Python 3.8+ installed
2. Clone this repository
3. Run the script:
   ```bash
   python mary_hangman_nv4.py
   ```

The script will automatically check for and install required packages if they're missing.

## Requirements

- Python 3.8+
- PyTorch 2.0.0+
- NumPy 1.21.0+
- scikit-learn 1.0.0+
- tqdm 4.65.0+

## Optional Arguments

- `--force-new-data`: Force generation of new training data
- `--evaluate MODEL_PATH`: Evaluate a saved model

## Data

The script expects the following files in the working directory:
- words_250000_train.txt: Training word list
- words_test.txt: Test word list 