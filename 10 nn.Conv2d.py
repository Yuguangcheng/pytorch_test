import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR100('dataset_Crfar10',train = False,transform=torchvision.transforms.ToTensor()
                                        ,download=True)
dataloader = DataLoader(dataset,batch_size=64)
writer = SummaryWriter('logs')

class Process(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = Conv2d(in_channels=3,out_channels=72,kernel_size=3,stride=1)#定义卷积层
    def forward(self,x):
        x = self.conv2d(x)
        return x

img_process = Process()
step = 0
for data in dataloader:
    imgs,targets = data
    output = img_process(imgs)

    output = torch.reshape(output, (-1, 3, 30,30))
    writer.add_images('input imgs',imgs,step)
    writer.add_images('output imgs', output, step)
    step = step+1
writer.close()