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

TO_CREATE = sys.argv[1]
TRAIN_HALF_SIZE = 5000
DEV_HALF_SIZE = 500
CHAR_EMB_SIZE = RNN_OUTPUT_SIZE = 30
NUM_OF_CHARS = 6
HIDDEN_LAYER = 24
OUTPUT_SIZE = 2
EPOCHS = 8
VOCAB = "()"
FILE_NAME = "./parentheses"



def create_dataset(TRAIN_HALF_SIZE=TRAIN_HALF_SIZE, DEV_HALF_SIZE=DEV_HALF_SIZE, file_name=FILE_NAME):

    def create(file_name, set_size):
        positive, negative = generate_positive_and_negative(set_size, type='parenthesis')

        positive, negative = {item + ' 1' for item in positive}, {item + ' 0' for item in negative}

        dataset = list(positive | negative)
        random.shuffle(dataset)
        write_to_file(file_name, dataset)

    create(file_name + "_train", TRAIN_HALF_SIZE)
    create(file_name + "_dev", DEV_HALF_SIZE)

def read_dataset(file_name=FILE_NAME):

    def read(file):
        with open(file, "r", encoding="utf-8") as f:
            data = f.readlines()
        seqs, tags = [], []
        for line in data:
            seq, tag = line.strip().split()
            seqs.append(seq)
            tags.append(tag)
        return seqs, tags

    train_x, train_y = read(file_name + "_train")
    dev_x, dev_y = read(file_name + "_dev")

    return train_x, train_y, dev_x, dev_y



class RNNAcceptorTagger(nn.Module):

    def __init__(self, char_emb_size=CHAR_EMB_SIZE, num_of_chars=NUM_OF_CHARS, rnn_output_size=RNN_OUTPUT_SIZE
                 , hidden_layer_size=HIDDEN_LAYER, output_size=OUTPUT_SIZE):
        super().__init__()

        self.rnn_output_size = rnn_output_size

        self.embeddings = nn.Embedding(char_emb_size, char_emb_size)

        self.lstm = nn.LSTM(char_emb_size, self.rnn_output_size)

        self.linear1 = nn.Linear(self.rnn_output_size, hidden_layer_size)

        self.linear2 = nn.Linear(hidden_layer_size, output_size)

        self.activation = nn.ReLU()
        self.optimizer = optim.Adam(self.parameters())
    def forward(self, seq):
        embs = self.embeddings(seq)

        lstm_out, _ = self.lstm(embs.view(len(seq), 1, -1))

        x = self.activation(self.linear1(lstm_out[-1].view(1, -1)))

        x = self.linear2(x)

        return F.log_softmax(x, dim=-1)
def accuracy_on_dataset(model, seqs, tags):
    model.eval()

    correct = total = 0.0
    sum_loss = 0.0

    with torch.no_grad():


        zipped = list(zip(seqs, tags))
        np.random.shuffle(zipped)
        x, y = zip(*zipped)

        for seq, tag in zip(x, y):
            total += 1
            if torch.cuda.is_available():
                seq = seq.cuda()
                tag = tag.cuda()

            seq = Variable(seq)

            output = model(seq)

            output = output.detach().cpu()
            tag = tag.cpu()

            y_hat = np.argmax(output.data.numpy())

            loss = F.nll_loss(output[0], tag)
            sum_loss += loss.item()

            if y_hat == tag:
                correct += 1

    return correct / total, sum_loss / len(tags)


def train(model, train_x, train_y, dev_x, dev_y):
    print("epoch", "train_loss", "train_accuracy", "dev_loss", "dev_accuracy", "passed_time")
    for epoch in range(EPOCHS):
        model.train()
        start_time = time()
        zipped = list(zip(train_x, train_y))
        np.random.shuffle(zipped)
        train_x, train_y = zip(*zipped)

        sum_loss = 0.0
        for seq, tag in zip(train_x, train_y):

            model.zero_grad()
            if torch.cuda.is_available():
                seq = seq.cuda()
                tag = tag.cuda()

            seq = Variable(seq)

            output = model(seq)

            loss = F.nll_loss(output[0], tag)
            sum_loss += loss.item()

            loss.backward()

            model.optimizer.step()

        train_loss = sum_loss / len(train_y)

        train_accuracy, _ = accuracy_on_dataset(model, train_x, train_y)

        dev_accuracy, dev_loss = accuracy_on_dataset(model, dev_x, dev_y)

        passed_time = time() - start_time

        print(epoch, train_loss, train_accuracy, dev_loss, dev_accuracy, passed_time)

if __name__ == "__main__":
    char_to_ix = {word: i for i, word in enumerate(sorted(list(VOCAB)))}
    ix_to_char = {i: word for i, word in enumerate(sorted(list(VOCAB)))}
    tag_to_ix = {"0": 0, "1": 1}
    if TO_CREATE == "1":
        create_dataset()
    train_x, train_y, dev_x, dev_y = read_dataset()
    train_x = [torch.tensor([char_to_ix[c] for c in seq]) for seq in train_x]
    train_y = [torch.tensor(tag_to_ix[t]) for t in train_y]
    dev_x = [torch.tensor([char_to_ix[c] for c in seq]) for seq in dev_x]
    dev_y = [torch.tensor(tag_to_ix[t]) for t in dev_y]

    tagger = RNNAcceptorTagger()
    train(tagger, train_x, train_y, dev_x, dev_y)
