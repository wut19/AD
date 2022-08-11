import pickle
import torch.utils.data
import numpy as np
import dlutils

class ReviewDataset:
    @staticmethod
    def pairs_separate(data):
        reviews = [pair[1] for pair in data]
        labels = [pair[0] for pair in data]
        return reviews,labels

    def __init__(self, data):
        self.reviews, self.labels = ReviewDataset.pairs_separate(data)

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


def make_datasets(data_path='./dataset/textdata.pkl'):
    input = open(data_path,'rb')
    data = pickle.load(input)
    print(data.keys())
    true_rv = []
    fake_rv = []
    for class1,items in data.items():
        if class1 == 1:
            true_rv += [(class1,item) for item in items]
        else :
            fake_rv += [(class1,item) for item in items]
            
    l1,l2 = int(len(true_rv)/5),int(len(fake_rv)/5)
    l = min(l1,l2)
    train_data = true_rv[:3*l1]  + fake_rv[:3*l2]
    # val_data = true_rv[3*l1:4*l1] + fake_rv[3*l2:4*l2]
    # test_data = true_rv[4*l1:] + fake_rv[4*l2:]
    val_data = true_rv[3*l1:3*l1+l] + fake_rv[3*l2:3*l2+l]
    test_data = true_rv[4*l1:4*l1+l] + fake_rv[4*l2:4*l2+l]
    
    train_set = ReviewDataset(train_data)
    val_set = ReviewDataset(val_data)
    test_set = ReviewDataset(test_data)
    print("dataset is ready!")
    
    return train_set,val_set,test_set

def make_dataloader(dataset,batch_size,device):
    # class BatchCollator(object):
    #     def __init__(self, device):
    #         self.device = device

    #     def __call__(self, batch):
    #         labels, reviews = batch
    #         return labels, reviews

    # data_loader = dlutils.batch_provider(dataset, batch_size, BatchCollator(device))
    data_loader = dlutils.batch_provider(dataset, batch_size)
    return data_loader    



 