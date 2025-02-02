import pandas as pd
import numpy as np
import os
import random
import string
import pickle
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import logging
from datetime import datetime

# Set up logging
def setup_logging():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = 'logs'
    Path(log_dir).mkdir(exist_ok=True)
    log_file = f'{log_dir}/bilstm_model_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

# Constants
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
DATA_DIR = 'hangman_data'
Path(DATA_DIR).mkdir(exist_ok=True)

def ensure_directories():
    """Create necessary directories if they don't exist"""
    directories = ['logs', DATA_DIR]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logging.info(f"Ensured directory exists: {directory}")

def train_loop(data_loader, model, loss_fn, optimizer, loss_estimate, batch_no, epoch, epoch_no):
    size = len(data_loader.dataset)
    model.train()
    for batch, (X, y) in enumerate(data_loader):
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()                
        if batch % 1000 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            loss_estimate.append(loss)
            batch_no.append(current)
            epoch_no.append(epoch)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(data_loader, model, loss_fn):
    size = len(data_loader.dataset)
    model.eval()
    num_batches = len(data_loader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for (X, y) in data_loader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(0) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

class CustomDatasetTrain(Dataset):
    def __init__(self, X_train, y_train):
        self.features = X_train
        self.label = y_train
        
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        features = self.features[idx]
        label = self.label[idx]
        sample = {"features": features, "label": label}
        return features, label

class extract_tensor(nn.Module):
    def forward(self,x):
        # Output shape (batch, features, hidden)
        tensor, _ = x
        # Reshape shape (batch, hidden)
        return tensor[:, -1, :]


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.LSTM_stack = nn.Sequential(
            nn.Embedding(64, 32, max_norm=1, norm_type=2),
            nn.LSTM(input_size=32, hidden_size=64, num_layers=1, batch_first=True, dropout=0.2, bidirectional=True),
            extract_tensor(),
            nn.Linear(128, 26)
        )
        # Move model to appropriate device
        self.to(DEVICE)
    
    def forward(self, x):
        logits = self.LSTM_stack(x)
        return logits

def create_dataloader(input_tensor, target_tensor):
    # Move tensors to appropriate device
    input_tensor = input_tensor.to(DEVICE)
    target_tensor = target_tensor.to(DEVICE)
    
    # Optimize batch size for MPS
    batch_size = 256 if torch.backends.mps.is_available() else 128
    
    all_features_data = CustomDatasetTrain(input_tensor, target_tensor)
    all_features_dataloader = DataLoader(all_features_data, batch_size=batch_size, shuffle=True)
    return all_features_dataloader

def save_model(model):
    model_dir = Path(DATA_DIR)
    model_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = model_dir / f"bilstm_model_{timestamp}.pt"
    torch.save(model.state_dict(), model_path)
    logging.info(f"Model saved to: {model_path}")

def train_model(input_tensor, target_tensor):
    # Clear MPS memory before training
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        
    all_features_dataloader = create_dataloader(input_tensor, target_tensor)
    model = NeuralNetwork()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # Changed from SGD to Adam
    
    loss_estimate = []
    batch_no = []
    epoch_no = []
    epochs = 8
    
    try:
        for t in range(epochs):
            logging.info(f"Epoch {t+1}\n-------------------------------")
            train_loop(all_features_dataloader, model, loss_fn, optimizer, loss_estimate, batch_no, t, epoch_no)
            test_loop(all_features_dataloader, model, loss_fn)
            
            # Save model after each epoch
            save_model(model)
            
        logging.info("Training completed!")
        
    finally:
        # Clear MPS memory after training
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

def get_char_mapping():
    char_mapping = {'_': 27}
    ct = 1
    for i in list(string.ascii_lowercase):
        char_mapping[i] = ct
        ct = ct + 1
    return char_mapping

def create_intermediate_data(df):
    """Create and return intermediate data frame with word features"""
    x = pd.DataFrame(df.split('\n'))
    x[1] = x[0].apply(lambda p: len(p))
    x['vowels_present'] = x[0].apply(lambda p: set(p).intersection({'a', 'e', 'i', 'o', 'u'}))
    x['vowels_count'] = x['vowels_present'].apply(lambda p: len(p))
    x['unique_char_count'] = x[0].apply(lambda p: len(set(p)))
    # Filter and return the dataframe
    return x[~((x['unique_char_count'].isin([0, 1, 2])) | (x[1] <= 3)) & (x.vowels_count != 0)]

def read_data():
    """Read training data with error handling"""
    file_path = Path("words_250000_train.txt")
    if not file_path.exists():
        raise FileNotFoundError("Training data file not found: words_250000_train.txt")
        
    with open(file_path, "r") as f:
        df = f.read()
    return df

def loop_for_permutation(unique_letters, word, all_perm, i):
    random_letters = random.sample(unique_letters, i+1)
    new_permuted_word = word
    for letter in random_letters:
        new_permuted_word = new_permuted_word.replace(letter, "_")
        all_perm.append(new_permuted_word)

def permute_all(word, vowel_permutation_loop=False):
    unique_letters = list(set(word))
    all_perm = []
    if vowel_permutation_loop:
        for i in range(len(unique_letters) - 1):
            loop_for_permutation(unique_letters, word, all_perm, i)
        all_perm = list(set(all_perm))
        return all_perm
    else:
        for i in range(len(unique_letters) - 2):
            loop_for_permutation(unique_letters, word, all_perm, i)
        all_perm = list(set(all_perm))
        return all_perm

def permute_consonents(word):
    len_word = len(word)
    vowel_word = "".join([i if i in ["a", "e", "i", "o", "u"] else "_" for i in list(word)])
    vowel_idxs = []
    for i in range(len(vowel_word)):
        if vowel_word[i] == "_":
            continue
        else:
            vowel_idxs.append(i)  
    abridged_vowel_word = vowel_word.replace("_", "")
    all_permute_consonents = permute_all(abridged_vowel_word, vowel_permutation_loop=True)
    permuted_consonents = []
    for permuted_word in all_permute_consonents:
        a = ["_"] * len(word)
        vowel_no = 0
        for vowel in permuted_word:
            a[vowel_idxs[vowel_no]] = vowel
            vowel_no += 1
        permuted_consonents.append("".join(a))
    return permuted_consonents

def create_masked_dictionary(df_aug):
    """Create dictionary of masked words"""
    masked_dictionary = {}
    counter = 0
    for word in df_aug[0]:
        all_masked_words_for_word = []
        all_masked_words_for_word = all_masked_words_for_word + permute_all(word)
        all_masked_words_for_word = all_masked_words_for_word + permute_consonents(word)
        all_masked_words_for_word = list(set(all_masked_words_for_word))
        masked_dictionary[word] = all_masked_words_for_word
        if counter % 10000 == 0:
            logging.info(f"Iteration {counter} completed")
        counter = counter + 1
    return masked_dictionary

def get_vowel_prob(df_vowel, vowel):
    try:
        return df_vowel[0].apply(lambda p: vowel in p).value_counts(normalize=True).loc[True]
    except:
        return 0

def get_vowel_prior(df_aug):
    prior_json = {}
    for word_len in range(df_aug[1].max()):
        prior_json[word_len + 1] = []
        df_vowel = df_aug[df_aug[1] == word_len]
        for vowel in ['a', 'e', 'i', 'o', 'u']:
            prior_json[word_len + 1].append(get_vowel_prob(df_vowel, vowel))
        prior_json[word_len + 1] = pd.DataFrame([pd.Series(['a', 'e', 'i', 'o', 'u']), pd.Series(prior_json[word_len + 1])]).T.sort_values(by=1, ascending=False)
    return prior_json    

def save_vowel_prior(vowel_prior):
    """Save vowel prior to the correct directory"""
    data_dir = Path(DATA_DIR)
    data_dir.mkdir(exist_ok=True)
    prior_path = data_dir / "prior_probabilities.pkl"
    
    with open(prior_path, "wb") as f:
        pickle.dump(vowel_prior, f)
    logging.info(f"Vowel prior saved to: {prior_path}")

def encode_output(word):
    char_mapping = get_char_mapping()
    output_vector = [0] * 26
    for letter in word:
        output_vector[char_mapping[letter] - 1] = 1
#     return torch.tensor([output_vector])
    return output_vector

def encode_input(word):
    char_mapping = get_char_mapping()
    given_word_len = len(word)
    embedding_len = 35
    word_vector = [0] * embedding_len
    ct = 0
    for letter_no in range(embedding_len - given_word_len, embedding_len):
        word_vector[letter_no] = char_mapping[word[ct]]
        ct += 1
    return word_vector

def encode_words(masked_dictionary): 
    target_data = []
    input_data = []
    counter = 0
    for output_word, input_words in masked_dictionary.items():
        output_vector = encode_output(output_word)
        for input_word in input_words:
            target_data.append(output_vector)
            input_data.append(encode_input(input_word))
        if counter % 10000 == 0:
            print(f"Iteration {counter} completed")
        counter += 1
    return input_data, target_data

def save_input_output_data(input_data, target_data):
    """Save data to the correct directory"""
    data_dir = Path(DATA_DIR)
    data_dir.mkdir(exist_ok=True)
    
    input_path = data_dir / 'input_features.txt'
    target_path = data_dir / 'target_features.txt'
    
    with open(input_path, 'w') as fp:
        for item in input_data:
            fp.write("%s\n" % item)
    logging.info(f'Input features saved to: {input_path}')
    
    with open(target_path, 'w') as fp:
        for item in target_data:
            fp.write("%s\n" % item)
    logging.info(f'Target features saved to: {target_path}')

def convert_to_tensor(input_data, target_data):
    input_tensor = torch.tensor(input_data, dtype=torch.long)
    target_tensor = torch.tensor(target_data, dtype=torch.float32)
    return input_tensor, target_tensor

def get_datasets():
    """Get or create training datasets"""
    try:
        # Read raw data
        df = read_data()
        logging.info("Data file read successfully")
        
        # Create intermediate data
        x_ = create_intermediate_data(df)
        if x_ is None or len(x_) == 0:
            raise ValueError("No valid data after intermediate processing")
        logging.info(f"Created intermediate data with {len(x_)} entries")
        
        # Create masked dictionary
        df_aug = x_.copy()
        masked_dictionary = create_masked_dictionary(df_aug)
        if not masked_dictionary:
            raise ValueError("Failed to create masked dictionary")
        logging.info(f"Created masked dictionary with {len(masked_dictionary)} entries")
        
        # Generate vowel prior probabilities
        vowel_prior = get_vowel_prior(df_aug)
        save_vowel_prior(vowel_prior)
        logging.info("Saved vowel prior probabilities")
        
        # Encode words
        input_data, target_data = encode_words(masked_dictionary)
        if not input_data or not target_data:
            raise ValueError("Failed to encode words")
        logging.info(f"Encoded {len(input_data)} input/target pairs")
        
        # Save encoded data
        save_input_output_data(input_data, target_data)
        
        # Convert to tensors
        input_tensor, target_tensor = convert_to_tensor(input_data, target_data)
        logging.info("Successfully created input and target tensors")
        
        return input_tensor, target_tensor
        
    except Exception as e:
        logging.error(f"Dataset creation failed: {str(e)}")
        raise

def main():
    log_file = setup_logging()
    logging.info(f"Using device: {DEVICE}")
    
    try:
        ensure_directories()
        input_tensor, target_tensor = get_datasets()
        train_model(input_tensor, target_tensor)
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise
    finally:
        logging.info(f"Log file saved to: {log_file}")

if __name__ == '__main__':
    main()

