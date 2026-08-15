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
def get_dataloader():
    my_dataset = MyPairsDataset(my_pairs)
    my_dataloder = DataLoader(my_dataset, batch_size=1, shuffle=True)

    # for i, (x,y) in enumerate(my_dataloder):
    #     print(f'the {i} batch data: {x, y}')
    #     break
    return my_dataloder

# TODO 6. Build the GRU encoder.
class EncoderGRU(nn.Module):
    # TODO 6.1 Define the initialization method.
    def __init__(self, input_size, hidden_size):
        """
        :param input_size: Input dimension of the encoder's word embedding layer,
                           i.e., the vocabulary size (2803 English words).
        :param hidden_size: Dimension of the encoder's hidden layer, 256.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 3. Instantiate the word embedding layer.
        # Input: [batch_size, seq_len] -> [batch_size, seq_len, hidden_size]
        self.embedding = nn.Embedding(input_size, hidden_size)

        # 4. Instantiate the GRU layer.
        # Arg 1: hidden_size: Input feature dimension,  the word embedding dimension.
        # Arg 2: hidden_size: Hidden state dimension, 256.
        # Arg 3: batch_first: Use the format：[batch_size, seq_len, hidden_size] -> [batch size, sequence length, embedding dimension].
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    # TODO 6.2 Define the forward pass.
    def forward(self, input, hidden):
        """
        Forward pass.
        :param input: Input word index sequence, [batch_size, seq_len] -> [1, 8].
        :param hidden: Initial hidden state, [num_layer, batch_size, hidden_size] -> [1, 1, 256].
        :return:
        """
        # 1. Pass the word index sequence through the embedding layer to convert word indices into word vectors.
        # Input shape: [batch_size, seq_len] -> [1, 8]
        # Output shape: [batch_size, seq_len, hidden_size] -> [1, 8, 256]
        output = self.embedding(input)

        # 2. Process the input through the GRU layer.
        # Input:
        #   output: Current input, i.e., [batch_size, seq_len, input_size] -> [1, 8, 256]
        #   hidden: Initial hidden state, i.e., [num_layer, batch_size, hidden_size] -> [1, 1, 256]
        # Output:
        #   output: Current output, i.e., [batch_size, seq_len, hidden_size] -> [1, 8, 256]
        #   hidden: Updated hidden state, i.e., [num_layer, batch_size, hidden_size] -> [1, 1, 256]
        output, hidden = self.gru(output, hidden)

        # 3. Return the GRU output and the final hidden state.
        return output, hidden

    # TODO 6.3 Extension: Define a custom method to initialize the hidden state, i.e., obtain h0.
    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)  # [num_layer, batch_size, hidden_size] -> [1, 1, 256]

# TODO 7. Build a GRU-based decoder -> Version 1: Without an attention mechanism.
class DecoderGRU(nn.Module):
    # TODO 7.1 Define the initialization method.
    def __init__(self, output_size, hidden_size):
        """
        Initialize the model attributes.
        :param output_size: Output dimension, i.e., the number of French words, 4345.
        :param hidden_size: Hidden layer dimension, i.e., 256.
        """
        # 1. Initialize the parent class.
        super().__init__()
        # 2. Save the input parameters.
        self.output_size = output_size
        self.hidden_size = hidden_size
        # 3. Create the word embedding layer.
        # Input: [batch_size, seq_len], Output: [batch_size, seq_len, hidden_size]
        self.embedding = nn.Embedding(output_size, hidden_size)
        # 4. Create the GRU layer.
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        # 5. Create the linear layer.
        # Input: [1, hidden_size], Output: [1, output_size]
        self.out = nn.Linear(hidden_size, output_size)
        # 6. Create the softmax layer -> Obtain the probability distribution.
        # dim=-1 means normalizing along the last dimension (the vocabulary dimension).
        self.softmax = nn.LogSoftmax(dim=-1)


    # TODO 7.2 Define the forward pass.
    def forward(self, input, hidden):
        # 1. Process the input through the word embedding layer.
        output = self.embedding(input)
        # 2. Apply the ReLU activation function.
        output = F.relu(output)
        # 3. Process the input through the GRU layer.
        # Input:
        #   output: Current input, i.e., [batch_size, 1, hidden_size] -> [1, 1, 256]
        #           The decoder generates the translation one word at a time.
        #   hidden: Previous hidden state, i.e., [num_layer, batch_size, hidden_size] -> [1, 1, 256]
        # Output:
        #   output: Current output, i.e., [batch_size, 1, hidden_size] -> [1, 1, 256]
        #   hidden: Updated hidden state, i.e., [num_layer, batch_size, hidden_size] -> [1, 1, 256]
        output, hidden = self.gru(output, hidden)

        # 3. Process the output through the linear and softmax layers.
        output = self.softmax(self.out(output[0]))

        # 4. Return the output and hidden state.
        return output, hidden


    # TODO 7.3 Extension: Define a custom method to initialize the hidden state, i.e., obtain h0.
    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

# TODO 8. Test the GRU-based decoder -> Version 1: Without Attention.
def dm_test_decoder():
    # 1. Get the data loader.
    my_dataloader = get_dataloader()
    # 2. Initialize the encoder model and move it to the GPU.
    my_encoder_gru = EncoderGRU(input_size=english_word_n, hidden_size=256).to(device)
    print(f'my_encoder_gru: {my_encoder_gru}')

    # 3. Initialize the decoder model and move it to the GPU.
    my_decoder_gru = DecoderGRU(output_size=french_word_n, hidden_size=256).to(device)
    print(f'my_decoder_gru: {my_decoder_gru}')

    # 4. Test the complete encoding -> decoding process.
    # 4.1 Get one batch of data (one sample) from the data loader.
    for i, (x, y) in enumerate(my_dataloader):
        # 4.2 Print information about the input data.
        print(f'Input data information (English sentence): {x.shape}, {x}')      # [1, 8]
        print(f'Input data information (French sentence): {y.shape}, {y}')      # [1, 6]

        # 4.3 Encoding process: Encode the English sentence into a sequence of hidden states.
        # Initialize the encoder.
        h0 = my_encoder_gru.init_hidden()
        # Encoder forward pass.
        encoder_output_c, hidden = my_encoder_gru(x, h0)
        print(f'Encoder output: {encoder_output_c.shape}')       # Shape: [1, 8, 256]

        # 4.4 Decoding process: Decode the hidden state sequence into a French sentence.
        # print(f'Observe: Output of the last time step: {encoder_output_c[0][-1].shape}, {encoder_output_c[0][-1]}') # [8, 256] -> last one [256]

        # 4.5 Specific decoding process -> Generate the translation one word at a time.
        # 4.5.1 Iterate over each time step of the target sentence.
        for i in range(y.shape[1]):
            # 4.5.2 Extract the target word index at the current time step.
            # y[0][i]: Get the index of the i-th word from the first sample in the batch.
            # view(1, -1): Convert the scalar into a [1, 1] tensor to match the decoder input requirements.
            tmp = y[0][i].view(1, -1)
            # 4.5.3 Perform the decoder forward pass.
            output, hidden = my_decoder_gru(tmp, hidden)
            # Print information about the decoder output.
            print(f'Probability distribution generated at each decoding time step: {output.size()}, {output.shape}')
        print('\n' * 5)

        break


if __name__ == '__main__':
    # test data processing function
    english_word2index, english_index2word, english_word_n, french_word2index, french_index2word, french_word_n, my_pairs = my_getdata()
    # print(f'English word-to-index mapping: {english_word2index}')
    # print(f'English index-to-word mapping: {english_index2word}')
    # print(f'Number of English words: {english_word_n}')
    # print(f'French word-to-index mapping: {french_word2index}')
    # print(f'French index-to-word mapping: {french_index2word}')
    # print(f'Number of French words: {french_word_n}')
    get_dataloader()
    dm_test_decoder()