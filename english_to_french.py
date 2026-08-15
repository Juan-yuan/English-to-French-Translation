import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torch.optim as optim
import time
import random
import matplotlib.pyplot as plt
import torch

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

# todo 1. Specify special Tokens
# Start marker
SOS_token = 0
# End marker
EOS_token = 1
# The maximum sentence length cannot exceed 10 tokens (including punctuation)
MAX_LENGTH = 10
data_path = './data/eng-fra-v2.txt'

# todo 2. Define the data cleaning function -> i.e., string normalization function, used for: text preprocessing.
def normalizeString(s):
    """
    String normalization function
    :param s: The string to be processed
    :return:
    """
    # 1. Convert the string to lowercase and remove leading and trailing whitespace.
    s = s.lower().strip()

    # 2. Add a space before .!? using regular expressions.
    # Param 1: Regular expression (i.e., the content to be replaced), Param 2: replacement content, Param 3: the string to operate on.
    s = re.sub(r'([.?!])', r' \1', s)

    # 3. Filter non-standard characters -> keep uppercase and lowercase letters and basic punctuation, replace other characters with spaces.
    s = re.sub('[^a-zA-Z.!?]+', r' ', s)

    # 4. Return the processed string.
    return s

# todo 3. Data preprocessing -> Clean the text and build the vocabulary dictionaries.
def my_getdata():
    # 1. Read the original file data.
    with open(data_path, 'r', encoding='utf-8') as src_f:
        # 1.1 Read all lines, obtaining: ['line 1\n', 'line 2\n'...]
        lines = src_f.readlines()

        # 2. Clean the text and build bilingual sentence pairs.
        my_pairs = [[normalizeString(s) for s in line.split('\t')] for line in lines]        # [['English sentence from line 1', 'French sentence from line 1'], ['English sentence from line 2', 'French sentence from line 2'],...]
        print(f'Total number of sentence pairs: {len(my_pairs)}')        # Total: 63594

        # 3. Initialize the English vocabulary.
        # 3.1 Create a dictionary mapping words to indices.
        english_word2index = {'SOS': 0, 'EOS': 1}
        # english_word2index = {'SOS': SOS_token, 'EOS': EOS_token}

        # 3.2 Initialize the English vocabulary size counter.
        english_word_n = 2

        # 4. Initialize the French vocabulary.
        french_word2index = {'SOS': 0, 'EOS': 1}
        french_word_n = 2

        # 5. Build the English vocabulary.
        # 5.1 Iterate through all bilingual sentence pairs and get the words from the English sentences.
        for pair in my_pairs:
            # 5.2 Process each English sentence to get the English words.
            for word in pair[0].split(' '):
                # 5.3 Check whether the word exists in the vocabulary. If not, add it and assign a new index.
                if word not in english_word2index:
                    english_word2index[word] = english_word_n       # Example: {'SOS': 0, 'EOS': 1, 'i': 2, ...}
                    english_word_n += 1

            # 5.3 Build the French vocabulary.
            for word in pair[1].split(' '):
                if word not in french_word2index:
                    french_word2index[word] = french_word_n
                    french_word_n += 1

        # 6. Build the reverse mappings, i.e., mappings from indices to words.
        # 6.1 English index-to-word mapping.
        english_index2word = {v: k for k, v in english_word2index.items()}
        # 6.2 French index-to-word mapping.
        french_index2word = {v: k for k, v in french_word2index.items()}

        # 7. Print vocabulary statistics.
        print(f'English vocabulary size: {english_word_n}')     # 2803
        print(f'French vocabulary size: {french_word_n}')      # 4345

        # 8. Return: English word-to-index mapping, index-to-word mapping, total number of words,
        # French word-to-index mapping, index-to-word mapping, total number of words, and bilingual sentence pairs.
        return english_word2index, english_index2word, english_word_n, french_word2index, french_index2word, french_word_n, my_pairs

# todo 4. Data preprocessing -> Build the Dataset object.
# 1. Call the my_getdata() function to retrieve the preprocessed data.
english_word2index, english_index2word, english_word_n, french_word2index, french_index2word, french_word_n, my_pairs = my_getdata()

# 2. Define MyPairsDataset, a custom dataset class.
class MyPairsDataset(Dataset):
    # todo 2.1 Initialization function.
    def __init__(self, my_pairs):
        self.my_pairs = my_pairs                # Sentence pairs, formatted as: [['English sentence from line 1', 'French sentence from line 1'], ['English sentence from line 2', 'French sentence from line 2'],...]
        self.sample_len = len(self.my_pairs)    # Number of sentence pairs.

    # todo 2.2 Method to get the total number of samples.
    def __len__(self):
        return self.sample_len

    # todo 2.3 Method to retrieve a sample at the specified index.
    def __getitem__(self, index):
        # 1. Adjust the index to ensure it is within the valid range. The index cannot be less than 0 or greater than the total number of samples - 1.
        index = min(max(index, 0), self.sample_len - 1)
        # 2. Retrieve the bilingual sentence pair by index. x represents the English sentence, and y represents the French sentence.
        x, y = self.my_pairs[index]
        # 3. Convert the English sentence text into numerical values.
        # 3.1 Split the sentence into words by spaces and get the index of each word.    Purpose: word -> corresponding word index -> word embedding.
        x = [english_word2index[word] for word in x.split(' ')]
        # 3.2 Append the end-of-sentence marker.
        x.append(EOS_token)
        # 3.3 Convert the list into a Tensor and specify the device.
        tensor_x = torch.tensor(x, dtype=torch.long, device=device)

        # 4. Convert the French sentence text into numerical values.
        y = [french_word2index[word] for word in y.split(' ')]
        y.append(EOS_token)
        tensor_y = torch.tensor(y, dtype=torch.long, device=device)

        # 5. Return the processed sample data.
        return tensor_x, tensor_y

# todo 5. Data procissing -> DataLoader
def get_dataloder():
    my_dataset = MyPairsDataset(my_pairs)
    my_dataloder = DataLoader(my_dataset, batch_size=1, shuffle=True)

    # for i, (x,y) in enumerate(my_dataloder):
    #     print(f'the {i} batch data: {x, y}')
    #     break
    return my_dataloder

if __name__ == '__main__':
    # test data processing function
    english_word2index, english_index2word, english_word_n, french_word2index, french_index2word, french_word_n, my_pairs = my_getdata()
    # print(f'English word-to-index mapping: {english_word2index}')
    # print(f'English index-to-word mapping: {english_index2word}')
    # print(f'Number of English words: {english_word_n}')
    # print(f'French word-to-index mapping: {french_word2index}')
    # print(f'French index-to-word mapping: {french_index2word}')
    # print(f'Number of French words: {french_word_n}')
    get_dataloder()