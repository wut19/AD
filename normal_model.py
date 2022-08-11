import torch 
import torch.nn as nn
import numpy as np

class Model(nn.Module):
    def __init__(self, n_vocab,embedd,sen_size,dropout,device):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(n_vocab, embedd, padding_idx=0)
        self.position_embedding = Positional_Encoding(embedd, sen_size, dropout, device)
        self.conv1 = nn.Conv1d(embedd,256,8,4,2)
        self.act = nn.ReLU()
        self.maxpool = nn.MaxPool1d(4,4,1)  
        self.conv2 = nn.Conv1d(256,512,3,2,1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.3)
        self.batchlayer = nn.BatchNorm1d(7*512)
        self.fc1 = nn.Linear(7*512,256)
        self.fc2 = nn.Linear(256,1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.position_embedding(self.embedding(x))
        #print(x.shape)
        x = torch.transpose(x,1,2)
        #print(x.shape)
        x = self.act(self.conv1(x))
        #print(x.shape)
        x= self.maxpool(x)
        #print(x.shape)
        x = self.batchlayer(self.dropout(self.flatten(self.conv2(x))))
        #print(x.shape)
        x = self.act(self.fc1(x))
        #print(x.shape)
        x = self.fc2(x)
        #print(x.shape)
        x = self.sigmoid(x)
        return x
            

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