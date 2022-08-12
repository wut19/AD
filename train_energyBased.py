# from transformer_model import Model
# from normal_model import Model
from textcnn_model import Model
from onehot_dataload_labelled import make_datasets,make_dataloader
import pickle
import os
import torch
from torch import optim
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


def train():
    n_vocab = 167657
    embedd = 300
    d_model = 200
    nhead = 4
    num_encoder = 2
    dropout = 0.5
    batch_size = 32
    lr = 5e-4
    epochs = 50
    sen_size = 200
    filters = [2,3,4,5]
    filters_number = 100
    l2_weight = 0.004
    m_in = -25
    m_out = -7
    
    device = torch.device('cpu')
    
    train_set_true,train_set_fake,eval_set,_ = make_datasets()
    
    #model = Model(n_vocab=n_vocab, embedd=embedd, sen_size=sen_size,d_model=d_model,nhead=nhead,num_encoder=num_encoder,dropout=dropout,device=device)
    #model = Model(n_vocab,embedd,sen_size,dropout,device)
    model = Model(n_vocab,embedd,sen_size,dropout,filters,filters_number,device)
    
    # model.load_state_dict(torch.load(os.path.join("models/transformer.pkl")))
    optimizer = optim.Adam(model.parameters(),lr=lr)
    loss_func = nn.CrossEntropyLoss()
    
    os.makedirs('models', exist_ok=True)
    

    for epoch in range(epochs):
        model.train()
        
        train_loader_true = make_dataloader(train_set_true,batch_size,device)
        train_loader_fake = make_dataloader(train_set_fake,batch_size,device)
        train_set_true.shuffle()
        train_set_fake.shuffle()
        
        if (epoch + 1) % 10 == 0:
            optimizer.param_groups[0]['lr'] /= 4
        count = 0
        loss_c = []
        for in_set,out_set in zip(train_loader_true,train_loader_fake):
            # print(x.shape)
            model.zero_grad()
            # y,x = torch.cat((1-y.reshape(-1,1),(y.reshape(-1,1))),dim=1).float(),x.long()
            y,x = torch.cat((in_set[0],out_set[0]),0),torch.cat((in_set[1],out_set[1]),0)
            #print(y.shape,x.shape)
            #print(x)
            y,x = (y>0).long(),x.long()
            # print(y.shape)
            y_p = model(x)
            # print(y_p.shape)
            l2_loss = l2_weight * torch.sum(torch.pow(list(model.parameters())[1], 2))
            Ec_out = -torch.logsumexp(y_p[batch_size:],dim=1)
            Ec_in = -torch.logsumexp(y_p[:batch_size],dim=1)
            # print(Ec_in,Ec_out)
            loss_E = 0.01*(torch.pow(F.relu(Ec_in-m_in), 2).mean() + torch.pow(F.relu(m_out-Ec_out), 2).mean())
            loss = loss_func(y_p,y) + l2_loss + loss_E
            loss_c.append(loss.item())
            count += 1
            plt.figure(figsize=(12, 8))
            plt.plot(np.arange(count),np.array(loss_c))
            plt.savefig(os.path.join('plot%d.png')%epoch)
            plt.close()
            loss.backward()
            optimizer.step()
        
        model.eval()
        eval_loader = make_dataloader(eval_set,batch_size=batch_size,device=device)
        eval_set.shuffle()
        n_count = 0
        r_count = 0
        with torch.no_grad():
            for y, x in eval_loader:
                # y,x = torch.cat(1-y.reshape(-1,1),(y.reshape(-1,1)),dim=1).float(),x.long()
                y,x = (y>0).long(),x.long()
                y_p = model(x)
                #print(em)
                #print(y_p.squeeze())
                y_p = (-torch.logsumexp(y_p,dim=1) <-16).long()
                
                #print(y_p.squeeze(),y)
                r_count += torch.sum(y_p.squeeze()==y)
                n_count += y.shape[0]
        #print(r_count,n_count)
        with open('accuracy.txt','a+') as f:
            f.write("epoch:%d accuracy:%f \n"%(epoch,r_count/n_count))
        torch.save(model.state_dict(),"models/conv_model%d.pkl"%epoch)

if __name__=="__main__":
    train()
        
            