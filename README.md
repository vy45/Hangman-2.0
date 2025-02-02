# Mary Hangman NV5

A deep learning model for playing Hangman using a transformer architecture.

## Prerequisites

- Python 3.8 or higher
- For Apple Silicon Macs: macOS 12.3 or higher

## Setup

1. Clone this repository
2. Ensure you have the required data files:
   - words_250000_train.txt
   - words_test.txt
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Model

```bash
python mary_hangman_nv5.py
```

## Optional Arguments

- `--force-new-data`: Force generation of new training data
- `--evaluate MODEL_PATH`: Evaluate a saved model
- `--load-weights PATH`: Load existing model weights
- `--latest-weights`: Load the most recent model weights

## Directory Structure

The script will create:
- `logs/`: Training logs
- `hangman_data/`: Model checkpoints and data files

## Data

The script expects the following files in the working directory:
- words_250000_train.txt: Training word list
- words_test.txt: Test word list 