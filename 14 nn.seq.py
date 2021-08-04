import torch
import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR100('dataset_Crfar10',train = False,transform=torchvision.transforms.ToTensor()
                                        ,download=True)
dataloader = DataLoader(dataset,batch_size=64)
class Process(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequential = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )
    def forward(self,x):
        x = self.sequential(x)
        return x
process = Process()
for data in dataloader:
    imgs,targets = data
    output = process(imgs)
    print(output.shape)