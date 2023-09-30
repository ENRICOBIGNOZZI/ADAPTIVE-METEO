# -*- coding: utf-8 -*-
"""SEQTOSEQ_MODEL.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1317sRmCAotDxeJl-bZ7NrlNtZuPp8Zn4
"""

class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim, n_features):
        super(Decoder, self).__init__()

        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features =  input_dim, n_features

        self.rnn1 = nn.LSTM(
          input_size=n_features,
          hidden_size=input_dim,
          num_layers=1,
          batch_first=True,
          dropout = 0
        )

        self.output_layer = nn.Linear(self.hidden_dim, n_features)
    def forward(self, x,input_hidden,input_cell):

        x, (hidden_n, cell_n) = self.rnn1(x,(input_hidden,input_cell))

        x = self.output_layer(x)
        return x, hidden_n, cell_n