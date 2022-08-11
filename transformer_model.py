import torch 
import torch.nn as nn
import numpy as np

class Model(nn.Module):
    def __init__(self, n_vocab,embedd,sen_size,d_model,nhead,num_encoder,dropout,device):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(n_vocab, embedd, padding_idx=0)
        self.position_embedding = Positional_Encoding(embedd, sen_size, dropout, device)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,dim_feedforward=1024),num_encoder)
        self.fc1 = nn.Linear(in_features=sen_size * d_model,out_features=1)
        self.act = nn.Sigmoid()
    def forward(self, x):
        x = self.embedding(x)
        x = self.position_embedding(x)
        # print(x.shape)
        x = torch.transpose(x,0,1)
        x = self.encoder(x)
        x = torch.transpose(x,0,1)
        print(x.shape)
        x = x.reshape(x.shape[0],-1)
        out = self.act(self.fc1(x))
        print(out.shape)
        return out
        

class Positional_Encoding(nn.Module):
    def __init__(self, embed, pad_size, dropout, device):
        super(Positional_Encoding, self).__init__()
        self.device = device
        self.pe = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = x + nn.Parameter(self.pe, requires_grad=False).to(self.device)
        out = self.dropout(out)
        return out