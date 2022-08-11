import pickle
import torch.utils.data
import numpy as np
import dlutils

class OnehotDataset:
    @staticmethod
    def pairs_separate(data):
        reviews = [pair[1][:200] for pair in data]
        labels = [pair[0] for pair in data]
        reviews,labels = np.array(reviews),np.array(labels)
        # print(reviews.shape)
        return reviews,labels

    def __init__(self, data):
        self.reviews, self.labels = OnehotDataset.pairs_separate(data)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.labels[index.start:index.stop], self.reviews[index.start:index.stop]
        return self.labels[index], self.reviews[index]

    def __len__(self):
        return len(self.labels)

    def shuffle(self):
        permutation = np.random.permutation(self.labels.shape[0])
        for x in [self.labels, self.reviews]:
            np.take(x, permutation, axis=0, out=x)

def make_datasets(data_path='./dataset/onehot_data.pkl'):
    input = open(data_path,'rb')
    true_onehot,fake_onehot = pickle.load(input)
            
    l1,l2 = int(len(true_onehot)/5),int(len(fake_onehot)/5)
    l = min(l1,l2)
    train_data = true_onehot[:3*l1]  + fake_onehot[:3*l2]
    # val_data = true_rv[3*l1:4*l1] + fake_rv[3*l2:4*l2]
    # test_data = true_rv[4*l1:] + fake_rv[4*l2:]
    val_data = true_onehot[3*l1:3*l1+l] + fake_onehot[3*l2:3*l2+l]
    test_data = true_onehot[4*l1:4*l1+l] + fake_onehot[4*l2:4*l2+l]
    
    train_set = OnehotDataset(train_data)
    val_set = OnehotDataset(val_data)
    test_set = OnehotDataset(test_data)
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