import torch
from torch import nn
import string
import random
import sys
import unidecode
from torch.utils.tensorboard import SummaryWriter

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Get characters from string.printable
all_characters = string.printable
n_characters = len(all_characters)

# read large text file
file = unidecode.unidecode(open("names.txt").read())

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embed = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)


    def forward(self, x, hidden, cell):
        out = self.embed(x)


class Generator():
    pass