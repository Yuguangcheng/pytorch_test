import torchvision
import torch
from torch import nn
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR100('dataset_Crfar10',train=False,transform=torchvision.transforms.ToTensor()
                                        ,download=True)
dataloader = DataLoader(dataset,batch_size=64,drop_last=True)
class linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(196608,10)
    def forward(self,input):
        output = self.linear(input)
        return output
L = linear()
for data in dataloader:
    imgs,targets = data
    print(imgs.shape)
    input = torch.flatten(imgs)#摊平 output = torch.reshape(imgs,(1,1,1,-1))
    output = L(input)
    print(output.shape)