import torch
from torch import nn
from torch.nn import functional as F

class Generator(nn.Module):
    def __init__(self, z_size, out_size):
        super(Generator, self).__init__()
        self.linear1 = nn.Linear(z_size, out_size)
        self.linear1_bn = nn.BatchNorm1d(out_size)
        self.linear2 = nn.Linear(out_size,2*out_size)
        self.linear2_bn = nn.BatchNorm1d(2*out_size)
        self.linear3 = nn.Linear(2*out_size,2*out_size)
        self.linear3_bn = nn.BatchNorm1d(2*out_size)
        self.linear4 = nn.Linear(2*out_size,out_size)


    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, x):
        x = F.leaky_relu(self.linear1_bn(self.linear1(x)),0.2)
        x = F.leaky_relu(self.linear2_bn(self.linear2(x)),0.2)
        x = F.leaky_relu(self.linear3_bn(self.linear3(x)),0.2)
        x = torch.tanh(self.linear4(x)) # 这里的激活函数不知道如何取,tanh需要将数据归一化
        return x
    

class Discriminator(nn.Module):
    def __init__(self, out_size):
        super(Discriminator, self).__init__()
        self.linear1 = nn.Linear(out_size,out_size//2)
        self.linear1_bn = nn.BatchNorm1d(out_size//2)
        self.linear2 = nn.Linear(out_size//2,out_size//4)
        self.linear2_bn = nn.BatchNorm1d(out_size//4)
        self.linear3 = nn.Linear(out_size//4,out_size//8)
        self.linear3_bn = nn.BatchNorm1d(out_size//8)
        self.linear4 = nn.Linear(out_size//8,1)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input):
        x = F.leaky_relu(self.linear1_bn(self.linear1(input)), 0.2)
        x = F.leaky_relu(self.linear2_bn(self.linear2(x)), 0.2)
        x = F.leaky_relu(self.linear3_bn(self.linear3(x)), 0.2)
        x = torch.sigmoid(self.linear4(x))
        return x
    

class Encoder(nn.Module):
    def __init__(self, z_size, input_size):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(input_size, input_size//2)
        self.linear1_bn = nn.BatchNorm1d(input_size//2)
        self.linear2 = nn.Linear(input_size//2,input_size//4)
        self.linear2_bn = nn.BatchNorm1d(input_size//4)
        self.linear3 = nn.Linear(input_size//4,input_size//8)
        self.linear3_bn = nn.BatchNorm1d(input_size//8)
        self.linear4 = nn.Linear(input_size//8,z_size)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, x):
        x = F.leaky_relu(self.linear1_bn(self.linear1(x)),0.2)
        x = F.leaky_relu(self.linear2_bn(self.linear2(x)),0.2)
        x = F.leaky_relu(self.linear3_bn(self.linear3(x)),0.2)
        x = self.linear4(x) 
        return x    


class ZDiscriminator(nn.Module):
    def __init__(self, z_size, batchSize, d=128):
        super(ZDiscriminator, self).__init__()
        self.linear1 = nn.Linear(z_size, d)
        self.linear2 = nn.Linear(d, d)
        self.linear3 = nn.Linear(d, 1)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, x):
        x = F.leaky_relu((self.linear1(x)), 0.2)
        x = F.leaky_relu((self.linear2(x)), 0.2)
        x = torch.sigmoid(self.linear3(x))
        return x


class ZDiscriminator_mergebatch(nn.Module):
    def __init__(self, z_size, batchSize, d=128):
        super(ZDiscriminator_mergebatch, self).__init__()
        self.linear1 = nn.Linear(z_size, d)
        self.linear2 = nn.Linear(d * batchSize, d)
        self.linear3 = nn.Linear(d, 1)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, x):
        x = F.leaky_relu((self.linear1(x)), 0.2).view(1, -1)  # after the second layer all samples are concatenated
        x = F.leaky_relu((self.linear2(x)), 0.2)
        x = torch.sigmoid(self.linear3(x))
        return x


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()