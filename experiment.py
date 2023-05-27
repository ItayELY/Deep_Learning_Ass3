import string
import sys
from time import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
import random
from gen_examples import write_to_file, generate_positive_and_negative

CHAR_EMB_SIZE=RNN_OUTPUT_SIZE=30
NUM_OF_CHARS=14
HIDDEN_LAYER=24
OUTTPUT_SIZE=2



def create_data_set(file_name, train_size, dev_size):

    def create(file_name, size):

        # Create the data set.
        positive, negative = generate_positive_and_negative(size)

        # Making the data set supervised.
        positive, negative = {item + ' 1' for item in positive}, {item + ' 0' for item in negative}

        data_set = list(positive | negative)

        # Shuffle the examples in the data set.
        random.shuffle(data_set)
        write_to_file(file_name, data_set)
        # Write the data set to a file.
        # write_examples(file, data_set)

    create(file_name + "_train", train_size)
    create(file_name + "_dev", dev_size)



class RNNAcceptorTagger(nn.Module):

    def __init__(self, char_emb_size=CHAR_EMB_SIZE, num_of_chars=NUM_OF_CHARS, rnn_output_size=RNN_OUTPUT_SIZE
                 , hidden_layer_size=HIDDEN_LAYER, output_size=OUTTPUT_SIZE):
        super().__init__()

        self.rnn_output_size = rnn_output_size

        self.embeddings = nn.Embedding(char_emb_size, char_emb_size)

        self.lstm = nn.LSTM(char_emb_size, self.rnn_output_size)

        self.linear1 = nn.Linear(self.rnn_output_size, hidden_layer_size)

        self.linear2 = nn.Linear(hidden_layer_size, output_size)

        self.activation = nn.ReLU()

    def forward(self, seq):
        embs = self.embeddings(seq)

        lstm_out, _ = self.lstm(embs.view(len(seq), 1, -1))

        x = self.activation(self.linear1(lstm_out[-1].view(1, -1)))

        x = self.linear2(x)

        return F.log_softmax(x, dim=-1)

