import torch
a = torch.cat((1-torch.tensor([1,2,3]).reshape(3,1),torch.tensor([2,3,4]).reshape(3,1)),dim=1)
print(torch.max(a,1))