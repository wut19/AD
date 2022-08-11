import torch.utils.data
from net import *
from torch.autograd import Variable
from jacobian import compute_jacobian_autograd
import numpy as np
import os
import scipy.optimize
from Embedded_dataload import make_datasets, make_dataloader
from evaluation import get_f1, evaluate
from threshold_search import find_maximum
import scipy.stats
from gensim.models import word2vec
from scipy.special import loggamma

def r_pdf(x, bins, counts):
    if bins[0] < x < bins[-1]:
        i = np.digitize(x, bins) - 1
        return max(counts[i], 1e-308)
    if x < bins[0]:
        return max(counts[0] * x / bins[0], 1e-308)
    return 1e-308

def extract_statistics(train_set, E, G, batchSize=128):
    
    zlist = []
    rlist = []
    
    dataloader = make_dataloader(train_set, batchSize,torch.device("cpu"))
    
    for label, x in dataloader:
        z = E(x)
        recon_batch = G(z)
        z = z.squeeze()
        
        recon_batch = recon_batch.squeeze().cpu().detach().numpy()
        x = x.squeeze().cpu().detach().numpy()
        
        z = z.cpu().detach().numpy()
        
        for i in range(x.shape[0]):
            distance = np.linalg.norm(x[i]-recon_batch[i])
            rlist.append(distance)
        
        zlist.append(z)

    zlist = np.concatenate(zlist)
    
    counts, bin_edges = np.histogram(rlist, bins=30, normed=True)
    
    def fmin(func, x0, args, disp):
        x0 = [2.0, 0.0, 1.0]
        return scipy.optimize.fmin(func, x0, args, xtol=1e-12, ftol=1e-12, disp=0)
    
    gennorm_param = np.zeros([3,zlist.shape[1]])
    for  i in range(zlist.shape[1]):
        betta, loc, scale = scipy.stats.gennorm.fit(zlist[:,i], optimizer=fmin)
        gennorm_param[0,i] = betta
        gennorm_param[1,i] = loc
        gennorm_param[2,i] = scale
        
    return counts, bin_edges, gennorm_param

def main():
    
    batchSize=128
    mul=0.2
    z_size = 16
    input_size = 200
    out_size = 200
    device = torch.device('cpu')
    model = word2vec.Word2Vec.load('word2vec.model')
    path = "./dataset/textdata.pkl"
    
    train_set, val_set, test_set = make_datasets(model,path)
    
    train_set.shuffle()
    
    G = Generator(z_size,out_size)
    E = Encoder(z_size,input_size)
    
    G.load_state_dict(torch.load(os.path.join('./models/Gmodel.pkl')))
    E.load_state_dict(torch.load(os.path.join('./models/Emodel.pkl')))
    
    G.eval()
    E.eval()
    
    counts, bin_edges, gennorm_param = extract_statistics(train_set,E,G,batchSize)
    
    def run_novely_prediction_on_dataset(dataset):

        
        result = []
        gt_novel = []
        
        data_loader = make_dataloader(dataset,batchSize,device)
        
        include_jacobian = True
        
        N = (200 - z_size) * mul    # 为什么要乘mul
        logC = loggamma(N/2.0) - (N/2.0)*np.log(2.0*np.pi)
        
        def logPe_func(x):
            return logC - (N-1) * np.log(x) + np.log(r_pdf(x, bin_edges, counts))   # 为什么是N-1
        
        for label, x in data_loader:
            
            x = Variable(x.data, requires_grad=True)
            
            z = E(x)
            recon_batch = G(z)
            z = z.squeeze()
            
            if include_jacobian:
                J = compute_jacobian_autograd(x,z)
                J = J.cpu().numpy()
            
            z = z.cpu().detach().numpy()
            
            recon_batch = recon_batch.squeeze().cpu().detach().numpy()
            x = x.squeeze().cpu().detach().numpy()
            
            for i in range(x.shape[0]):
                if include_jacobian:
                    u,s,vh = np.linalg.svd(J[i,:,:],full_matrices=False)
                    # logD = -np.sum(np.log(np.abs(s)))   # 这里应该没有负号？
                    logD = np.sum(np.log(np.abs(s)))
                else:
                    logD = 0

                p = scipy.stats.gennorm.pdf(z[i],gennorm_param[0,:],gennorm_param[1,:],gennorm_param[2,:])
                logPz = np.sum(np.log(p))
                
                if not np.isfinite(logPz):
                    logPz = -1000
                
                distance = np.linalg.norm(x[i] - recon_batch[i])
                
                logPe = logPe_func(distance)
                
                P = logD + logPz + logPe
                
                result.append(P)
                gt_novel.append(label[i].item() == 1)
        
        result = np.asarray(result,dtype=np.float32)
        ground_truth = np.asarray(gt_novel,dtype=np.float32)
        
        return result, ground_truth
    
    def compute_threshold(val_set):
        y_scores, y_true = run_novely_prediction_on_dataset(val_set)
        
        minP = min(y_scores) - 1
        maxP = max(y_scores) + 1
        y_false = np.logical_not(y_true)
        
        def evaluate(e):
            y = np.greater(y_scores,e)
            tp = np.sum(np.logical_and(y,y_true))
            fp = np.sum(np.logical_and(y,y_false))
            fn = np.sum(np.logical_and(np.logical_not(y),y_true))
            return get_f1(tp,fp,fn)
        
        best_th, best_f1 = find_maximum(evaluate,minP,maxP,1e-4)
        
        return best_th
    
    def test(test_set,threshold):
        y_score, y_true = run_novely_prediction_on_dataset(test_set)
        return evaluate(y_score, y_true, threshold)
    
    e = compute_threshold(val_set)
    results = test(test_set,e)
    
    return results
                
    