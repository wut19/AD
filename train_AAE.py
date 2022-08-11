from torch import optim
import torch.utils.data
from Embedded_dataload import make_datasets,make_dataloader
from net import Generator, Discriminator, Encoder, ZDiscriminator, ZDiscriminator_mergebatch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import os
from tracker import LossTracker

def train(model,z_size=16,input_size=200,out_size=200,batchSize=128,d=128,cross_batch=False,lr=0.002,epoch_count=30):
    
    os.makedirs('models', exist_ok=True)
    
    train_set,_,_ = make_datasets(model,'./dataset/textdata.pkl')
    
    G = Generator(z_size,out_size)
    G.weight_init(0,0.02)
    
    D = Discriminator(out_size)
    D.weight_init(0,0.2)
    
    E = Encoder(z_size,input_size)
    E.weight_init(0,0.02)
    
    if cross_batch:
        ZD = ZDiscriminator_mergebatch(z_size,batchSize,d)
    else:
        ZD = ZDiscriminator(z_size,batchSize,d)
    ZD.weight_init(0,0.02)
    
    G_optimizer = optim.Adam(G.parameters(),lr=lr,betas=(0.5,0.999))
    D_optimizer = optim.Adam(D.parameters(),lr=lr,betas=(0.5,0.999))
    GE_optimizer = optim.Adam(list(E.parameters()) + list(G.parameters()), lr=lr, betas=(0.5, 0.999))
    ZD_optimizer = optim.Adam(ZD.parameters(), lr=lr, betas=(0.5, 0.999))
    
    BCE_loss = nn.BCELoss()
    sample = torch.randn(64,z_size).view(-1,z_size,1,1)
    
    output_folder = './tracker_results/'
    tracker = LossTracker(output_folder)
    
    for epoch in range(epoch_count):
        G.train()
        D.train()
        E.train()
        ZD.train()
        
        data_loader = make_dataloader(train_set,batchSize,torch.device("cpu"))
        train_set.shuffle()
        
        if (epoch + 1) % 10 == 0:
            G_optimizer.param_groups[0]['lr'] /= 4
            D_optimizer.param_groups[0]['lr'] /= 4
            GE_optimizer.param_groups[0]['lr'] /= 4
            ZD_optimizer.param_groups[0]['lr'] /= 4
            print("learning rate change!")
            
        for y, x in data_loader:
            y_real_ = torch.ones(x.shape[0])
            y_fake_ = torch.ones(x.shape[0])
            
            y_real_z = torch.ones(1 if cross_batch else x.shape[0])
            y_fake_z = torch.ones(1 if cross_batch else x.shape[0])
        
            ###########################################
            
            D.zero_grad()
            
            D_result = D(x).squeeze()
            D_real_loss = BCE_loss(D_result,y_real_)
            
            z = torch.randn((x.shape[0],z_size))
            z = Variable(z)
            # print(z.shape)
            x_fake = G(z).detach()
            D_result = D(x_fake).squeeze()
            D_fake_loss = BCE_loss(D_result,y_fake_)
            
            D_train_loss = D_real_loss + D_fake_loss
            D_train_loss.backward()
            
            D_optimizer.step()
            
            tracker.update(dict(D=D_train_loss))
            
            ##############################################
            
            G.zero_grad()
            
            z = torch.randn((x.shape[0], z_size)).view(-1, z_size)
            z = Variable(z)

            x_fake = G(z)
            D_result = D(x_fake).squeeze()

            G_train_loss = BCE_loss(D_result, y_real_)

            G_train_loss.backward()
            G_optimizer.step()

            tracker.update(dict(G=G_train_loss))
            
            ############################################
            
            ZD.zero_grad()

            z = torch.randn((x.shape[0], z_size)).view(-1, z_size)
            z = z.requires_grad_(True)

            ZD_result = ZD(z).squeeze()
            ZD_real_loss = BCE_loss(ZD_result, y_real_z)

            z = E(x).squeeze().detach()

            ZD_result = ZD(z).squeeze()
            ZD_fake_loss = BCE_loss(ZD_result, y_fake_z)

            ZD_train_loss = ZD_real_loss + ZD_fake_loss
            ZD_train_loss.backward()

            ZD_optimizer.step()

            tracker.update(dict(ZD=ZD_train_loss))
            
            #############################################
            
            E.zero_grad()
            G.zero_grad()
            
            z = E(x)
            x_d = G(z)
            
            ZD_result = ZD(z.squeeze()).squeeze()
            
            E_train_loss = BCE_loss(ZD_result,y_fake_z) * 1.0
            
            # Recon_loss = F.binary_cross_entropy(x_d,x.detach()) * 2.0   # 这里的损失函数应该如何选取
            Recon_loss = F.mse_loss(x_d,x.detach()) * 2.0
            
            (Recon_loss + E_train_loss).backward()
            
            GE_optimizer.step()
            
            tracker.update(dict(GE=Recon_loss, E=E_train_loss))
            
            ####################################################
        
        print("epoch %d"%epoch)
        tracker.register_means(epoch)
        tracker.plot()
        
    print("training finish!... saving the results")
    
    torch.save(G.state_dict(), "models/Gmodel.pkl")
    torch.save(E.state_dict(), "models/Emodel.pkl")
    #torch.save(D.state_dict(), "Dmodel_%d_%d.pkl" %(folding_id, ic))
    #torch.save(ZD.state_dict(), "ZDmodel_%d_%d.pkl" %(folding_id, ic))