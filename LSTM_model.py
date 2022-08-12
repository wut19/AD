import torch
from torch import nn

class Model(nn.Module):

  def __init__(self, n_vocab,embedd,sen_size,hidden_size,nhead,num_layers,dropout):
    super(Model, self).__init__()
    self.embedding = nn.Embedding(n_vocab, embedd)
    self.lstm      = nn.LSTM(
      input_size = embedd,
      hidden_size = hidden_size,
      num_layers = num_layers,
      bias = True,
      batch_first = True,
      dropout = 1.0, # dropout
      bidirectional = True
    )
    self.linear    = nn.Linear(
      in_features = hidden_size * 2,
      out_features = 2
    )
    # self.softmax   = nn.Softmax()

  def forward(self, input_data):
    """
    :param input_data: [batch_size, seq_length]
    :return:
    """
    # [batch_size, seq_length, embedding_size]
    output = self.embedding(input_data)
    # output [batch_size, seq_length, 2*hidden_size]
    output, _ = self.lstm(output)
    # [batch_size, 2*hidden_size]
    output = output[:, -1, :].squeeze(dim=1)
    # [batch_size, num_classes]
    output = self.linear(output)
    # output = self.softmax(output)
    return output