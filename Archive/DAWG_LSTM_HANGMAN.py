import numpy as np
from keras.layers import Conv1D, LSTM, Dense, TimeDistributed, Embedding, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.layers import Bidirectional
import random

# Load and preprocess data
def load_data(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        words = file.read().splitlines()
    return words

def simulate_missing_letters(word, p_missing=0.4):
    """Randomly replace letters with '0' to simulate missing letters"""
    simulated_word = ''
    for char in word:
        simulated_word += '0' if random.random() < p_missing else char
    return simulated_word

def preprocess_data(words, char_to_int, max_word_length, p_missing=0.4):
    sequences = []
    targets = []
    for word in words:
        # Create partially obscured word
        word_with_missing = simulate_missing_letters(word, p_missing)
        # Convert to integer sequences
        sequences.append([char_to_int[char] if char in char_to_int else char_to_int['0'] 
                        for char in word_with_missing])
        targets.append([char_to_int[char] for char in word])
    sequences = pad_sequences(sequences, maxlen=max_word_length, padding='post')
    targets = pad_sequences(targets, maxlen=max_word_length, padding='post')
    return sequences, targets

file_path = '/kaggle/input/english-words/words_250000_train.txt'
words = load_data(file_path)
chars = sorted(list(set(''.join(words) + '0')))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
max_word_length = 32

X, y = preprocess_data(words, char_to_int, max_word_length)
y = to_categorical(y, num_classes=len(chars))

model = Sequential()
model.add(Embedding(input_dim=len(chars), output_dim=128, trainable=True))
model.add(Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(Bidirectional(LSTM(256, return_sequences=True, kernel_regularizer=l2(0.001))))
model.add(Dropout(0.4))
model.add(TimeDistributed(Dense(len(chars), activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min', restore_best_weights=True)

model.fit(X, np.array(y), validation_split=0.2, epochs=20, batch_size=128, callbacks=[early_stopping])



from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

def predict_word(model, word_with_missing, char_to_int, int_to_char, max_word_length, guessed_letters):
    print("LSTM")
    word_encoded = [char_to_int[char] if char in char_to_int else char_to_int['0'] for char in word_with_missing]
    word_padded = pad_sequences([word_encoded], maxlen=max_word_length, padding='post')
    
    prediction = model.predict(word_padded)
    predict_vector = prediction[0]
    
    for i, char in enumerate(word_with_missing):
        if char == '0':
            probabilities = predict_vector[i]
            sorted_indices = np.argsort(-probabilities)
            for idx in sorted_indices:
                predicted_char = int_to_char[idx]
                if predicted_char != '0' and predicted_char not in guessed_letters:
                    return predicted_char
    return None  # Return None if all characters are guessed or no suitable character is found

word_with_missing = "an000"
guessed_letters = set(['e','o','i','s','t','d','a'])
predicted_char = predict_word(model, word_with_missing, char_to_int, int_to_char, max_word_length, guessed_letters)
print(predicted_char)

class Node:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class DAWG:
    def __init__(self):
        self.root = Node()

    def add_word(self, word):
        current = self.root
        for letter in word:
            if letter not in current.children:
                current.children[letter] = Node()
            current = current.children[letter]
        current.is_end_of_word = True
    
    # search for words with similar pattern using dfs
    def generate_possible_words(self, node, partial_word):
        possible_words = []

        def dfs(current_node, current_prefix):
            if current_node.is_end_of_word:
                possible_words.append(current_prefix)
            if len(current_prefix) < len(partial_word):
                next_letter_index = len(current_prefix)
                next_letter = partial_word[next_letter_index]
                for letter, child_node in current_node.children.items():
                    if next_letter == '_' or next_letter == letter:
                        dfs(child_node, current_prefix + letter)

        dfs(node, '')
        return possible_words

    def predict_next_letter(self, partial_word, guessed_letters):
        # Convert word to current game state using guessed letters
        word = partial_word.replace(" ", "")
        word = ''.join([letter if letter in guessed_letters else '_' for letter in word])
        
        # Early game strategy (DAWG-based)
        possible_words = self.generate_possible_words(self.root, partial_word)
        letter_frequency = {}
        for word in possible_words:
            for letter in word:
                # Only count frequencies of unguessed letters
                if letter not in letter_frequency and letter not in guessed_letters:
                    letter_frequency[letter] = 1
                elif letter not in guessed_letters:
                    letter_frequency[letter] += 1

        sorted_letters = sorted(letter_frequency.items(), key=lambda x: x[1], reverse=True)
        for letter, _ in sorted_letters:
            if letter not in partial_word:
                return letter
            
        # If no suitable letter is found, return a random letter not in guessed_letters
        word = partial_word.replace(" ", "")
        word = ''.join([letter if letter in guessed_letters else '_' for letter in word])
        word_with_missing = word.replace('_', '0')
        return predict_word(model, word_with_missing, char_to_int, int_to_char, max_word_length, guessed_letters)

hangman_dawg = DAWG()
for word in words:
    hangman_dawg.add_word(word)

    import json
import requests
import random
import string
import secrets
import time
import re
import collections
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

try:
    from urllib.parse import parse_qs, urlencode, urlparse
except ImportError:
    from urlparse import parse_qs, urlparse
    from urllib import urlencode

from requests.packages.urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

class HangmanAPI(object):
    def __init__(self, access_token=None, session=None, timeout=None):
        self.hangman_url = self.determine_hangman_url()
        self.access_token = access_token
        self.session = session or requests.Session()
        self.timeout = timeout
        self.guessed_letters = []
        
        full_dictionary_location = '/kaggle/input/english-words/words_250000_train.txt'
        self.full_dictionary = self.build_dictionary(full_dictionary_location)        
        self.full_dictionary_common_letter_sorted = collections.Counter("".join(self.full_dictionary)).most_common()
        
        self.current_dictionary = []
        
    @staticmethod
    def determine_hangman_url():
        links = ['https://trexsim.com', 'https://sg.trexsim.com']

        data = {link: 0 for link in links}

        for link in links:

            requests.get(link)

            for i in range(10):
                s = time.time()
                requests.get(link)
                data[link] = time.time() - s

        link = sorted(data.items(), key=lambda x: x[1])[0][0]
        link += '/trexsim/hangman'
        return link

    def guess(self, word):
        
        ###############################################
        # Replace with your own "guess" function here #
        ###############################################

        clean_word = word[::2]
        
        # find length of passed word
        len_word = len(clean_word)

        partial_word = clean_word
        next_letter = hangman_dawg.predict_next_letter(partial_word, self.guessed_letters)
        
        guess_letter = next_letter         
        
        return guess_letter

    ##########################################################
    # You'll likely not need to modify any of the code below #
    ##########################################################
    
    def build_dictionary(self, dictionary_file_location):
        text_file = open(dictionary_file_location,"r")
        full_dictionary = text_file.read().splitlines()
        text_file.close()
        return full_dictionary
                
    def start_game(self, practice=True, verbose=True):
        # reset guessed letters to empty set and current plausible dictionary to the full dictionary
        self.guessed_letters = []
        self.current_dictionary = self.full_dictionary
                         
        response = self.request("/new_game", {"practice":practice})
        if response.get('status')=="approved":
            game_id = response.get('game_id')
            word = response.get('word')
            tries_remains = response.get('tries_remains')
            if verbose:
                print("Successfully start a new game! Game ID: {0}. # of tries remaining: {1}. Word: {2}.".format(game_id, tries_remains, word))
            while tries_remains>0:
                # get guessed letter from user code
                guess_letter = self.guess(word)
                    
                # append guessed letter to guessed letters field in hangman object
                self.guessed_letters.append(guess_letter)
                if verbose:
                    print("Guessing letter: {0}".format(guess_letter))
                    
                try:    
                    res = self.request("/guess_letter", {"request":"guess_letter", "game_id":game_id, "letter":guess_letter})
                except HangmanAPIError:
                    print('HangmanAPIError exception caught on request.')
                    continue
                except Exception as e:
                    print('Other exception caught on request.')
                    raise e
               
                if verbose:
                    print("Sever response: {0}".format(res))
                status = res.get('status')
                tries_remains = res.get('tries_remains')
                if status=="success":
                    if verbose:
                        print("Successfully finished game: {0}".format(game_id))
                    return True
                elif status=="failed":
                    reason = res.get('reason', '# of tries exceeded!')
                    if verbose:
                        print("Failed game: {0}. Because of: {1}".format(game_id, reason))
                    return False
                elif status=="ongoing":
                    word = res.get('word')
        else:
            if verbose:
                print("Failed to start a new game")
        return status=="success"
        
    def my_status(self):
        return self.request("/my_status", {})
    
    def request(
            self, path, args=None, post_args=None, method=None):
        if args is None:
            args = dict()
        if post_args is not None:
            method = "POST"

        # Add `access_token` to post_args or args if it has not already been
        # included.
        if self.access_token:
            # If post_args exists, we assume that args either does not exists
            # or it does not need `access_token`.
            if post_args and "access_token" not in post_args:
                post_args["access_token"] = self.access_token
            elif "access_token" not in args:
                args["access_token"] = self.access_token

        time.sleep(0.2)

        num_retry, time_sleep = 50, 2
        for it in range(num_retry):
            try:
                response = self.session.request(
                    method or "GET",
                    self.hangman_url + path,
                    timeout=self.timeout,
                    params=args,
                    data=post_args,
                    verify=False
                )
                break
            except requests.HTTPError as e:
                response = json.loads(e.read())
                raise HangmanAPIError(response)
            except requests.exceptions.SSLError as e:
                if it + 1 == num_retry:
                    raise
                time.sleep(time_sleep)

        headers = response.headers
        if 'json' in headers['content-type']:
            result = response.json()
        elif "access_token" in parse_qs(response.text):
            query_str = parse_qs(response.text)
            if "access_token" in query_str:
                result = {"access_token": query_str["access_token"][0]}
                if "expires" in query_str:
                    result["expires"] = query_str["expires"][0]
            else:
                raise HangmanAPIError(response.json())
        else:
            raise HangmanAPIError('Maintype was not text, or querystring')

        if result and isinstance(result, dict) and result.get("error"):
            raise HangmanAPIError(result)
        return result
    
class HangmanAPIError(Exception):
    def __init__(self, result):
        self.result = result
        self.code = None
        try:
            self.type = result["error_code"]
        except (KeyError, TypeError):
            self.type = ""

        try:
            self.message = result["error_description"]
        except (KeyError, TypeError):
            try:
                self.message = result["error"]["message"]
                self.code = result["error"].get("code")
                if not self.type:
                    self.type = result["error"].get("type", "")
            except (KeyError, TypeError):
                try:
                    self.message = result["error_msg"]
                except (KeyError, TypeError):
                    self.message = result

        Exception.__init__(self, self.message)
api = HangmanAPI(access_token="api_key", timeout=2000)
for i in range(100):
    print('Playing ', i, ' th game')
    # Uncomment the following line to execute your final runs. Do not do this until you are satisfied with your submission
    api.start_game(practice=1,verbose=True)
    
    # DO NOT REMOVE as otherwise the server may lock you out for too high frequency of requests
    time.sleep(0.5)