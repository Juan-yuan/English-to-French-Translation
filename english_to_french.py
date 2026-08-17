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
from tqdm import tqdm

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

# todo 6. Build the GRU encoder.
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

# todo 7. Build a GRU-based decoder -> Version 1: Without attention mechanism.
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

# todo 8. Test the GRU-based decoder -> Version 1: Without Attention.
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

# todo 9. Build a GRU-based decoder -> Version 2: with Attention mechanism
class AttnDecoderGRU(nn.Module):
    # todo 9.1 Initialization function.
    def __init__(self, output_size, hidden_size, dropout_p=0.1, max_length=MAX_LENGTH):
        """
        Initialize the decoder attributes.
        :param output_size: Target (French) vocabulary size.
        :param hidden_size: Hidden layer dimension -> consistent with the encoder.
        :param dropout_p: Dropout probability -> helps prevent overfitting.
        :param max_length: Maximum sentence length -> limits the attention computation range.
        """
        # 1. Initialize the parent class.
        super().__init__()
        # 2. Store the input parameters.
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        # 3. Word embedding layer.
        # Input shape: [batch_size, seq_len] -> [1, 1]
        # Output shape: [batch_size, seq_len, hidden_size] -> [1, 1, 256]
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        # 4. Attention weight calculation layer.
        # Param 1: Concatenated query vector and hidden state -> 512
        # Param 2: Attention weight distribution -> up to 10 words
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        # 5. Attention fusion layer, combining the word embedding and attention context.
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        # 6. Dropout layer, randomly drops some neurons to help prevent overfitting.
        self.dropout = nn.Dropout(self.dropout_p)
        # 7. GRU layer.
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        # 8. Output layer: maps the GRU hidden state to the target vocabulary size.
        self.out = nn.Linear(self.hidden_size, self.output_size)
        # 9. LogSoftmax layer -> maps the output to a log-probability distribution.
        self.softmax = nn.LogSoftmax(dim=-1)


    # todo 9.2 Forward propagation.
    def forward(self, input, hidden, encoder_outputs):
        """
        Forward propagation function.
        :param input: Input word index at the current time step -> [batch_size, 1] -> [1, 1]
        :param hidden: Hidden state from the previous time step -> [1, batch_size, hidden_size] -> [1, 1, 256]
        :param encoder_outputs: Outputs from all encoder time steps -> [batch_size, seq_len, hidden_size] -> [1, 8, 256]
        :return:
        """
        # 1. Word embedding layer.
        # Input shape: [batch_size, 1] -> [1, 1], resulting in [1, 1, 256].
        embedded = self.embedding(input)
        # 2. Apply Dropout for regularization.
        embedded = self.dropout(embedded)
        # 3. Calculate the attention weights.
        # step1: torch.cat((embedded[0], hidden[0]), 1) -> [1, 512]
        # step2: self.attn(torch.cat((embedded[0], hidden[0]), 1)) -> map through a linear layer to the attention length -> [1, 10]
        # step3: Apply softmax() to convert the scores into a probability distribution -> [1, 10]
        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        # 4. Calculate the attention context -> the aggregated information from the encoder.
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        # 5. Attention fusion layer, combining the word embedding and attention context.
        output = torch.cat((embedded[0], attn_applied[0]), 1)       # [1, 1, 512]
        output = self.attn_combine(output).unsqueeze(0)             # [1, 1, 256]
        # 6. Apply the activation function to enhance the model's nonlinear capability.
        output = F.relu(output)     # Enhanced Q -> the input for the current time step.
        # 7. GRU layer: processes sequential data and maintains the hidden state.
        output, hidden = self.gru(output, hidden)
        # 8. Map the output to the target vocabulary size and apply LogSoftmax.
        output = self.softmax(self.out(output[0]))
        # 9. Return the results.
        # Param 1: Output probability distribution at the current time step.    [1, 4345]
        # Param 2: Updated hidden state (i.e., the hidden state for the current time step). [1, 1, 256]
        # Param 3: Attention weight distribution at the current time step, used for visualization. If visualization is not needed, it can be omitted.
        return output, hidden, attn_weights

    # todo 9.3 Extended_custom initialization function -> used to obtain h0.
    def init_hidden(self):
        # Shape: [1, 1, 256]
        return torch.zeros(1, 1, self.hidden_size, device=device)


# todo 10. Test the GRU-based decoder -> Test Version 2: with Attention mechanism
def dm_test_attn_decoder():
    # 1. Get the data loader.
    my_dataloader = get_dataloader()

    # 2. Model initialization.
    # 2.1 Create the encoder.
    # Param 1: English vocabulary size (2803), Param 2: Hidden size (256)
    my_encoder = EncoderGRU(english_word_n, 256).to(device)

    # 2.2 Create the decoder.
    # Param 1: French vocabulary size (4345), Param 2: Hidden size (256)
    my_decoder = AttnDecoderGRU(french_word_n, 256).to(device)

    # 3. Model training (inference) stage.
    # 3.1 Get one sample from the data loader.
    for i, (x, y) in enumerate(my_dataloader):
        # 3.2 Print the input information.
        print(f'x (English sentence): {x.shape}, {x}')
        print(f'y (French sentence): {y.shape}, {y}')

        # 3.3 Encoding process -> encode the English sentence into hidden states.
        # Initialize the encoder hidden state.
        hidden = my_encoder.init_hidden()

        # output: Hidden state at each time step -> [1, seq_len, 256]
        # hidden: Hidden state from the last time step -> [1, 1, 256]
        output, hidden = my_encoder(x, hidden)

        # 3.4 Prepare the encoder outputs for the Attention mechanism.
        # Create a fixed-size tensor to store the encoder outputs.
        # Shape: [10, 256]
        encoder_output_c = torch.zeros(
            MAX_LENGTH,
            my_encoder.hidden_size,
            device=device
        )

        # 3.5 Copy the actual encoder outputs into the fixed-size tensor.
        for idx in range(output.shape[1]):
            encoder_output_c[idx] = output[0, idx]

        # 3.6 Decoding process: decode the hidden states into a French sentence.
        # The decoder generates the translation one word at a time.
        # 3.6.1 Iterate through each time step of the target sentence.
        for i in range(y.shape[1]):

            # 3.6.2 Extract the target word index at the current time step.
            tmp = y[0][i].view(1, -1)

            # 3.6.3 Perform the decoder forward pass.
            # Parameter details:
            #   tmp: Current input word index -> [1, 1]
            #   hidden: Hidden state from the previous time step -> [1, 1, 256]
            #   encoder_output_c: Encoder outputs from all time steps -> [10, 256]

            # Return values:
            #   output: Output probability distribution at the current time step -> [1, 4345]
            #   hidden: Updated hidden state at the current time step -> [1, 1, 256]
            #   attn_weights: Attention weight distribution at the current time step -> [1, 10]
            output, hidden, attn_weights = my_decoder(
                tmp,
                hidden,
                encoder_output_c
            )

            # 3.6.4 Print the output shapes.
            print(f'decoder output.shape: {output.shape}')              # [1, 4345]
            print(f'decoder hidden.shape: {hidden.shape}')              # [1, 1, 256]
            print(f'decoder attn_weights.shape: {attn_weights.shape}')  # [1, 10]
            print('\n')

        break       # Test only one sample.


# todo 11. Build the model's internal iterative training function -> complete the training process for a single batch.
# Define the model training parameters.
# Learning rate, number of training epochs, Teacher Forcing ratio, output logging interval (print once every 1000 samples),
# and plotting interval (plot once every 100 samples).
my_lr, epochs, teacher_forcing_ratio, print_interval_num, plot_interval_num = 1e-4, 5, 0.5, 1000, 100

# Define the function to train a single batch: encoding -> decoding -> backpropagation -> parameter optimization...
def train_iters(x, y, my_encoder_rnn, my_attn_decoder_rnn, myadam_encode, myadam_decode, my_crossentropy_loss):
    """
    This function trains a single batch: encoding -> decoding -> backpropagation -> parameter optimization...
    :param x: input sequence -> [batch_size, seq_len], [1, seq_len]
    :param y: target sequence -> [batch_size, seq_len], [1, seq_len]
    :param my_encoder_rnn: encoder object
    :param my_attn_decoder_rnn: decoder object with attention mechanism
    :param myadam_encode: encoder optimizer
    :param myadam_decode: decoder optimizer
    :param my_crossentropy_loss: loss function
    :return:
    """
    # 1. Encoding stage -> convert the input sequence into a context representation. Initial hidden state: [1, 1, 256]
    encoder_hidden = my_encoder_rnn.init_hidden()
    encoder_output, encoder_hidden = my_encoder_rnn(x, encoder_hidden)

    # 2. Prepare decoder parameters.
    # 2.1 Build the encoder output tensor for attention calculation. Shape: [10, 256]
    encoder_output_c = torch.zeros(MAX_LENGTH, my_encoder_rnn.hidden_size, device=device)
    # Copy the actual encoder output into a fixed-length tensor: [assuming 6 words, 256] -> [10, 256]
    for idx in range(x.shape[1]):
        encoder_output_c[idx] = encoder_output[0, idx]

    # 2.2 Initialize the decoder hidden state.
    decoder_hidden = encoder_hidden     # Use the encoder's final hidden state as the decoder's initial hidden state.

    # 2.3 Initial decoder input.
    input_y = torch.tensor([[SOS_token]], device=device)  # [1, 1]

    # 3. Initialize the loss.
    my_loss = 0.0
    y_len = y.shape[1]      # Target sequence length (i.e., the length of the French sentence to predict), e.g., 8

    # 4. Decide whether to use Teacher Forcing based on probability.
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    if use_teacher_forcing:
        # 4.1 Teacher Forcing model: use the ground-truth label as the input for the next step.
        for i in range(y_len):
            # Input:
            #   input_y -> [1, 1]
            #   decoder_hidden -> [1, 1, 256]
            #   encoder_output_c -> [10, 256]
            # Output:
            #   output_y -> [1, 4345]
            #   decoder_hidden -> [1, 1, 256]
            #   attn_weights -> [1, 10]
            output_y, decoder_hidden, attn_weights = my_attn_decoder_rnn(input_y, decoder_hidden, encoder_output_c)
            # Get the ground-truth label for the current time step.
            target_y = y[0][i].view(1)
            # Accumulate the loss.
            my_loss += my_crossentropy_loss(output_y, target_y)
            # Use the ground-truth label directly as the input for the next time step.
            input_y = y[0][i].view(1, -1)
            # input_y = target_y            # Same effect as above.
    else:
        # 4.2 Non-Teacher Forcing model: use the predicted label from the previous time step as the input for the next step.
        for i in range(y_len):
            # Input:
            #   input_y -> [1, 1]
            #   decoder_hidden -> [1, 1, 256]
            #   encoder_output_c -> [10, 256]
            # Output:
            #   output_y -> [1, 4345]
            #   decoder_hidden -> [1, 1, 256]
            #   attn_weights -> [1, 10]
            output_y, decoder_hidden, attn_weights = my_attn_decoder_rnn(input_y, decoder_hidden, encoder_output_c)
            # Get the ground-truth label for the current time step.
            target_y = y[0][i].view(1)
            # Accumulate the loss.
            my_loss += my_crossentropy_loss(output_y, target_y)
            # Get the next predicted word, i.e., the index and probability of the word with the highest probability.
            topv, topi = output_y.topk(1)
            # Stop prediction if the end-of-sentence token is predicted.
            if topi.squeeze().item() == EOS_token:
                break
            # If the end-of-sentence token was not predicted, use the predicted word (label) as the input for the next time step.
            input_y = topi.detach()         # [1, 1]

    # 5. Backpropagation + parameter update.
    myadam_encode.zero_grad()       # Clear gradients for the encoder.
    myadam_decode.zero_grad()       # Clear gradients for the decoder.

    # Perform backpropagation and calculate gradients.
    my_loss.backward()
    # Update parameters.
    myadam_encode.step()
    myadam_decode.step()

    # 6. Return the average loss.
    return my_loss.item() / y_len

# todo 12. Build the model training parameters, initialize the models and optimizers, and train for multiple epochs.
def train_seq2seq():
    # 1. Get the data loader.
    my_dataloader = get_dataloader()
    # 2. Initialize the models and move them to the GPU.
    # 2.1 Encoder input dimension = English vocabulary size 2803, hidden dimension: 256.
    my_encoder_rnn = EncoderGRU(english_word_n, 256).to(device)
    # 2.2 Decoder input dimension = French vocabulary size 4345, hidden dimension: 256, dropout probability: 0.1, maximum sentence length.
    my_attn_decoder_rnn = AttnDecoderGRU(french_word_n, 256, dropout_p=0.1, max_length=MAX_LENGTH).to(device)
    # 3. Initialize the optimizers using Adam with a learning rate of 1e-4.
    myadam_encode = optim.Adam(my_encoder_rnn.parameters(), lr=my_lr)       # Encoder -> optimizer.
    myadam_decode = optim.Adam(my_attn_decoder_rnn.parameters(), lr=my_lr)  # Decoder -> optimizer.
    # 4. Initialize the loss function using NLLLoss (Negative Log-Likelihood Loss).
    my_crossentropy_loss = nn.NLLLoss()
    # 5. Initialize training parameters.
    plot_loss_list = []     # Store loss values for plotting.
    # 6. Start the training loop.
    # 6.1 Outer loop, controlling the number of training epochs.
    for epoch_idx in range(1, epochs + 1):
        # Initialize the loss accumulators for the current epoch.
        print_loss_total, plot_loss_total = 0.0, 0.0
        # Record the start time.
        start_time = time.time()

        # 6.2 Inner loop, iterate through each sample in the dataset (the actual training process for each epoch).
        # for item, (x, y) in enumerate(my_dataloader, start=1):
        for item, (x, y) in enumerate(tqdm(my_dataloader), start=1):
            # Call the internal training function to train a single sample.
            myloss = train_iters(x, y, my_encoder_rnn, my_attn_decoder_rnn, myadam_encode, myadam_decode, my_crossentropy_loss)
            # Accumulate the loss.
            print_loss_total += myloss
            plot_loss_total += myloss
            # 6.3 Print training logs -> print once every print_interval_num=1000 samples.
            if item % print_interval_num == 0:
                # Calculate the average loss.
                print_loss_avg = print_loss_total / print_interval_num
                # Reset the loss accumulator.
                print_loss_total = 0.0
                # Print training information: epoch, average loss, and elapsed time.
                print(f'Epoch: {epoch_idx}, Loss: {print_loss_avg:.4f}, Time: {(time.time() - start_time):.2f} s')

            # 6.4 Record the loss for plotting -> record once every plot_interval_num=100 samples.
            if item % plot_interval_num == 0:
                # Calculate the average loss.
                plot_loss_avg = plot_loss_total / plot_interval_num
                # Store the loss value.
                plot_loss_list.append(plot_loss_avg)
                # Reset the loss accumulator.
                plot_loss_total = 0.0

            # 6.5 Temporary extension: stop training after 3000 samples per epoch.
            # Remove this condition in the actual implementation.
            if item > 3000:
                break

        # 7. Reaching this point means one epoch has finished. Save the models.
        torch.save(my_encoder_rnn.state_dict(), f'./model/my_encoder_rnn_{epoch_idx}.pth')
        torch.save(my_attn_decoder_rnn.state_dict(), f'./model/my_attn_decoder_rnn_{epoch_idx}.pth')

    # 8. Reaching this point means training has finished. Plot the loss curve.
    plt.figure()
    plt.plot(plot_loss_list)
    plt.savefig('./img/seq2seq_loss.png')
    plt.show()

    # 9. Training has finished. Return the result.
    return plot_loss_total      # Training loss list -> average loss every 100 samples.


# todo 13. Build the model evaluation (testing) function and use the trained Seq2Seq model for translation.
# Param 1: Input English sentence index tensor -> [1, seq_len]
# Param 2: Model encoder object
# Param 3: Model decoder object
def seq2seq_evaluate(x, my_encoder_rnn, my_attn_decoder_rnn):
    # Extension: Disable gradient calculation to save memory and speed up inference.
    with torch.no_grad():
        # 1. Encoding stage -> convert the input English sentence into a hidden state.
        encode_hidden = my_encoder_rnn.init_hidden()
        encode_output, encode_hidden = my_encoder_rnn(x, encode_hidden)

        # 2. Prepare decoder parameters.
        # 2.1 Build a fixed-length encoder output tensor.
        encoder_output_c = torch.zeros(MAX_LENGTH, my_encoder_rnn.hidden_size, device=device)
        for idx in range(x.shape[1]):
            encoder_output_c[idx] = encode_output[0, idx]

        # 2.2 Initial decoder hidden state.
        decode_hidden = encode_hidden

        # 2.3 Initial decoder input: Start-of-sentence token.
        input_y = torch.tensor([[SOS_token]], device=device)

        # 3. Autoregressive decoding process -> generate the target sentence token by token.
        # Store the decoded French words.
        decode_words = []
        # Initialize the attention matrix.
        decoder_attentions = torch.zeros(MAX_LENGTH, MAX_LENGTH)

        # Start decoding.
        for idx in range(MAX_LENGTH):
            # 3.1 Decoder -> forward pass.
            # Input: current input word index, decoder hidden state, encoder output tensor.
            # Output: probability distribution for the next word, updated hidden state, and attention weights.
            output_y, decode_hidden, attn_weights = my_attn_decoder_rnn(input_y, decode_hidden, encoder_output_c)
            # 3.2 Record the attention weights.
            decoder_attentions[idx] = attn_weights
            # 3.3 Predict the next word.
            topv, topi = output_y.topk(1)       # Get the index and probability of the word with the highest probability.
            # 3.4 Handle the termination condition: stop generation if the EOS token is predicted.
            if topi.squeeze().item() == EOS_token:
                decode_words.append('<EOS>')
                break
            else:
                # 3.5 Otherwise, add the predicted word to the result list.
                decode_words.append(french_index2word[topi.squeeze().item()])

        # 3.6 Update the input: use the current predicted word as the input for the next time step.
        input_y = topi.detach()

    # Return the decoded result and attention matrix.
    # decode_words: List of decoded French words.
    # decoder_attentions: Attention weight matrix, [seq_len, MAX_LENGTH]
    return decode_words, decoder_attentions[:idx + 1]

# todo 14. Call the model evaluation function, load the trained models, and translate custom samples.
# Model paths.
PATH1 = './model/my_encoder_rnn_5.pth'
PATH2 = './model/my_attn_decoder_rnn_5.pth'

# Define the function to evaluate the model.
def dm_test_seq2seq_evaluate():
    # 1. Prepare the data.
    my_pairs_dataset = MyPairsDataset(my_pairs)
    my_dataloader = DataLoader(my_pairs_dataset, batch_size=1, shuffle=False)
    # 2. Load the encoder model.
    input_size = english_word_n
    hidden_size = 256
    my_encoder_rnn = EncoderGRU(input_size, hidden_size).to(device)
    # 3. Load the model weights.
    # map_location: Ensures the model can be loaded on both CPU and GPU (CUDA).
    # Normally, the model should be trained and evaluated on the same device.
    # Using map_location allows a model trained on GPU to be evaluated on CPU.
    # weights_only=True: Load only the model weights.
    my_encoder_rnn.load_state_dict(torch.load(PATH1, map_location='cpu', weights_only=True), False)
    print(f'my_encoder_rnn encoder model architecture: {my_encoder_rnn}')

    # 4. Load the decoder model.
    input_size = french_word_n
    my_attn_decoder_rnn = AttnDecoderGRU(input_size, hidden_size).to(device)
    my_attn_decoder_rnn.load_state_dict(torch.load(PATH2, map_location='cpu', weights_only=True), False)
    print(f'my_attn_decoder_rnn decoder model architecture: {my_attn_decoder_rnn}')

    # 5. Define custom test samples.
    my_sample_pairs = [
        # Format: [English sentence, French sentence]
        ['i m wet .', 'je suis mouillee .'],
        ['i m glad you came over .', 'je me rejouis que vous soyez venue .'],
        ['she s pleased with her new dress .', 'sa nouvelle robe lui plait .'],
    ]
    print(f'Custom test samples: {my_sample_pairs}')

    # 6. Translate each sample.
    for index, pair in enumerate(my_sample_pairs):
        x = pair[0]     # English sentence.
        y = pair[1]     # French sentence.

        # 6.1 Convert text to numerical values -> English sentence -> index list.
        tmpx = [english_word2index[word] for word in x.split(' ')]
        tmpx.append(EOS_token)      # Add the end-of-sentence token.
        tensor_x = torch.tensor(tmpx, dtype=torch.long, device=device).view(1, -1)

        # 6.2 Run model prediction.
        decode_words, attentions = seq2seq_evaluate(tensor_x, my_encoder_rnn, my_attn_decoder_rnn)
        # 6.3 Convert the predicted words into a sentence.
        output_sentence = ' '.join(decode_words)
        print(f'Input (original English sentence): {x}')
        print(f'Input (original French sentence): {y}')
        print(f'Output (predicted French sentence): {output_sentence}')
        print('-' * 30)


if __name__ == '__main__':
    # test data processing function
    english_word2index, english_index2word, english_word_n, french_word2index, french_index2word, french_word_n, my_pairs = my_getdata()
    # print(f'English word-to-index mapping: {english_word2index}')
    # print(f'English index-to-word mapping: {english_index2word}')
    # print(f'Number of English words: {english_word_n}')
    # print(f'French word-to-index mapping: {french_word2index}')
    # print(f'French index-to-word mapping: {french_index2word}')
    # print(f'Number of French words: {french_word_n}')
    # get_dataloader()
    # dm_test_decoder()
    # dm_test_attn_decoder()
    train_seq2seq()