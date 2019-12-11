import os
import time
import glob
import pickle
import pathlib

import numpy
import numpy as np

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from os.path import exists, join
from matplotlib import pyplot as plt
from music21 import converter, instrument, note, chord, stream

device = 'cuda'

DATA_DIR = "./data/"

midi_notes = []

for i, file in enumerate(glob.glob(os.path.join(DATA_DIR, "*.mid"))):
    midi_p = pathlib.Path(file)
    midi_file_name = midi_p.stem

    midi = converter.parse(file)
    print('\r', 'Parsing file ', i, " ", file, end='')

    notes_to_parse = midi.flat.notes

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            midi_notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            midi_notes.append('.'.join(str(n) for n in element.normalOrder))

with open(join(DATA_DIR, 'notes.pickle'), 'wb') as filepath:
    pickle.dump(midi_notes, filepath)

n_vocab = (len(set(midi_notes)))


def categor(y, num_classes=None, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


sequence_length = 100

pitch_names = sorted(set(item for item in midi_notes))

note_to_int = dict((note, number) for number, note in enumerate(pitch_names))
network_input = []
network_output = []
for i in range(0, len(midi_notes) - sequence_length, 1):
    sequence_in = midi_notes[i:i + sequence_length]
    sequence_out = midi_notes[i + sequence_length]
    is_end = False
    for j, n in enumerate(sequence_in):
        if n == 'end':
            i = j + 1
            is_end = True
            continue
        if is_end:
            sequence_in[j] = 'end'
    if is_end:
        sequence_out = 'end'

    network_input.append([note_to_int[char] for char in sequence_in])

    network_output.append(note_to_int[sequence_out])
n_patterns = len(network_input)

network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))

network_input = network_input / float(n_vocab)
network_output = np.array(network_output)


class LSTMGen(nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size, batch_size=1):
        super(LSTMGen, self).__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.l = nn.LSTM(input_dim, hidden_dim)
        self.l1 = nn.LSTM(hidden_dim, hidden_dim)
        self.l2 = nn.LSTM(hidden_dim, hidden_dim)
        self.hl1 = self.init_hidden()
        self.hl2 = self.init_hidden()
        self.h2l = nn.Linear(hidden_dim, 256)
        self.h2v = nn.Linear(256, n_vocab)

    def init_hidden(self):
        return (torch.zeros(1, self.batch_size, self.hidden_dim).to(device),
                torch.zeros(1, self.batch_size, self.hidden_dim).to(device))

    def forward(self, seq):
        lstm_out, self.hl1 = self.l(seq, self.hl1)
        lstm_out = F.dropout(lstm_out, 0.3)

        lstm1, self.hl2 = self.l1(lstm_out, self.hl2)

        linear_layer = self.h2l(lstm1)
        linear_layer = F.dropout(linear_layer, 0.3)
        linear_layer1 = self.h2v(linear_layer)

        return linear_layer1


from random import sample, seed, shuffle

seed(0)


def data_spliter(input_, gt, train_f=1., val_f=0.0, test_f=0.0):
    assert len(gt) == len(input_)
    index = list(range(input_.shape[0]))
    index_train = sample(index, int(input_.shape[0] * train_f))
    index_val_test = list(set(index) - set(index_train))
    index_val = sample(index, int(len(index_val_test) * val_f))
    index_test = list(set(index_val_test) - set(index_val))

    DATA_SET = {
        "train": [input_[index_train, ...], gt[index_train]],
        "val": [input_[index_val, ...], gt[index_val]],
        "test": [input_[index_test, ...], gt[index_test]]
    }
    return DATA_SET


DATA_SET = data_spliter(network_input, network_output)

train_in, train_gt = DATA_SET["train"]
val_in, val_gt = DATA_SET["val"]
test_in, test_gt = DATA_SET["test"]


def stb(data, batch_size, is_shuffle=False):
    data_in = data[0]
    data_gt = data[1]
    if is_shuffle:
        shuffle_indxs = list(range(len(data_in)))
        shuffle(shuffle_indxs)
        data_in = [data_in[idx] for idx in shuffle_indxs]
        data_gt = [data_gt[idx] for idx in shuffle_indxs]

    for batch_ind in range(batch_size, len(data_in), batch_size):
        if device == "cuda":
            yield torch.tensor(data_in[batch_ind - batch_size:batch_ind]).cuda(), torch.tensor(
                data_gt[batch_ind - batch_size:batch_ind]).cuda()
        else:
            yield torch.tensor(data_in[batch_ind - batch_size:batch_ind]), torch.tensor(
                data_gt[batch_ind - batch_size:batch_ind])


def init_weights(layer):
    if hasattr(layer, "weight"):
        init.xavier_normal(layer.weight)
        layer.bias.data.fill_(0.01)


input_dim = network_input.shape[1]

model = LSTMGen(100, 512, n_vocab, batch_size=64)
model.apply(init_weights)
model.to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=0.001)
loss_logger = []
loss_logger_val = []

it = 0
for epoch in range(500):
    model.train()
    loss_epoch = []
    print(epoch)

    for i, (batch_in, batch_gt) in enumerate(stb(DATA_SET["train"], 64, True)):
        it += 1
        model.hl1 = model.init_hidden()
        model.hl2 = model.init_hidden()

        prediction = model(batch_in.view(batch_in.shape[-1], batch_in.shape[0], -1).float())

        log_loss = loss_function(prediction.view(prediction.shape[1], -1), batch_gt.long())

        loss = log_loss.item()
        if it % 100 == 0 and i > 0:
            print("Epoch: {}, Iteration: {}, Loss: {}".format(epoch, it, loss))
            loss_epoch.append(loss)
        optimizer.zero_grad()
        log_loss.backward()
        optimizer.step()

    loss_logger.append(loss)

torch.save(model.state_dict(),
           'model.pth'.format(time=time.time(), epoch=epoch, loss=np.mean(loss_epoch)))

plt.plot(range(len(loss_logger)), loss_logger)
plt.xlabel("iter")
plt.ylabel("loss")

plt.legend(['Train'])
plt.show()
