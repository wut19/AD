import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    
    def __init__(self,n_vocab,embedd,sen_size,dropout,filters,filters_number,device):
        super(Model, self).__init__()

        # 词嵌入层
        self.embedding = nn.Embedding(n_vocab, embedd)

        #卷积层
        # if  torch.cuda.is_available():
        #     self.convs = [
        #         nn.Sequential(
        #             nn.Conv1d(embedd, filters_number, filter_size),
        #             nn.ReLU(),
        #             nn.MaxPool1d(kernel_size=sen_size - filter_size + 1)
        #         ).cuda()
        #         for filter_size in filters]
        # else:
        self.convs = [
            nn.Sequential(
                nn.Conv1d(embedd, filters_number, filter_size),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=sen_size - filter_size + 1)
            )
            for filter_size in filters]

        # # 卷积层
        # if config.use_cuda:
        #     self.convs = [nn.Conv1d(config.embedding_dim, config.filters_number, filter_size).cuda()
        #                   for filter_size in config.filters]
        # else:
        #     self.convs = [nn.Conv1d(config.embedding_dim, config.filters_number, filter_size)
        #                   for filter_size in config.filters]

        # 正则化处理
        self.dropout = nn.Dropout(dropout, inplace=True)

        # 分类层
        self.fc = nn.Linear(filters_number*len(filters), 2)

    def conv_and_pool(self,x,conv):
        x = F.relu(conv(x))
        # 池化层
        x = F.max_pool1d(x,x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2).contiguous()
        x = torch.cat([conv_relu_pool(x) for conv_relu_pool in self.convs], dim=1).squeeze(2)
        # x = torch.cat([self.conv_and_pool(x, conv) for conv in self.convs], 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x