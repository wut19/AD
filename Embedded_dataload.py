import pickle
import torch.utils.data
import numpy as np
import dlutils

class EmbeddedDataset:
    @staticmethod
    def embedding(data,model):
        reviews = [pair[1] for pair in data]
        labels = [pair[0] for pair in data]       
        # embedded_vectors = [np.mean(np.asarray([model.wv[word] for word in review if model.wv.has_index_for(word)],dtype='float'),axis=0) for review in reviews]
        embedded_vectors = []
        for review in reviews:
            embedded_review = []
            for word in review:
                if model.wv.has_index_for(word):
                    embedded_word = model.wv[word]
                    embedded_review.append(embedded_word)
            if len(embedded_review) > 0:
                # embedded_review = np.mean(np.asarray(embedded_review),axis=0)
                embedded_vectors.append(embedded_review)
            else:
                embedded_review = np.zeros_like(np.asarray(model.wv['happy']))
            # embedded_vectors.append(embedded_review)
        
        maxnorm = np.max(np.linalg.norm(model.wv.vectors, axis=1))
        return np.asarray(labels), np.asarray(embedded_vectors),np.float32(maxnorm)
    def __init__(self, data, model):
        self.y, self.x, self.maxnorm = EmbeddedDataset.embedding(data,model)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.y[index.start:index.stop], self.x[index.start:index.stop]/self.maxnorm
        return self.y[index], self.x[index]/self.maxnorm

    def __len__(self):
        return len(self.y)

    def shuffle(self):
        permutation = np.random.permutation(self.y.shape[0])
        for x in [self.y, self.x]:
            np.take(x, permutation, axis=0, out=x)

def make_datasets(model,data_path='./dataset/textdata.pkl'):
    input = open(data_path,'rb')
    data = pickle.load(input)
    # print(data.keys())
    true_rv = []
    fake_rv = []
    for class1,items in data.items():
        if class1 == 1:
            true_rv += [(class1,item) for item in items]
        else :
            fake_rv += [(class1,item) for item in items]
            
    l1,l2 = int(len(true_rv)/5),int(len(fake_rv)/5)
    l = min(l1,l2)
    train_data = true_rv[:3*l1] + fake_rv[:3*l2]
    # val_data = true_rv[3*l1:4*l1] + fake_rv[3*l2:4*l2]
    # test_data = true_rv[4*l1:] + fake_rv[4*l2:]
    val_data = true_rv[3*l1:3*l1+l] + fake_rv[3*l2:3*l2+l]
    test_data = true_rv[4*l1:4*l1+l] + fake_rv[4*l2:4*l2+l]
    
    train_set = EmbeddedDataset(train_data,model)
    val_set = EmbeddedDataset(val_data,model)
    test_set = EmbeddedDataset(test_data,model)
    print("dataset is ready!")
    
    return train_set,val_set,test_set

def make_dataloader(dataset, batch_size, device):
    class BatchCollator(object):
        def __init__(self, device):
            self.device = device

        def __call__(self, batch):
            with torch.no_grad():
                y, x = batch
                x = torch.tensor(x, requires_grad=True, dtype=torch.float32, device=self.device)
                y = torch.tensor(y, dtype=torch.int32, device=self.device)
                return y, x

    data_loader = dlutils.batch_provider(dataset, batch_size, BatchCollator(device))
    return data_loader