#最大池化
import torch
from torch import nn
from torch.nn import MaxPool2d
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR100('dataset_Crfar10',train=False,transform=torchvision.transforms.ToTensor(),download=True)
writer = SummaryWriter('logs')
input = torch.tensor([[1,2,3,4,5],
                      [4,6,7,8,9],
                      [4,6,7,8,9],
                      [4,6,7,8,9],
                      [4,6,7,8,9]],dtype=torch.float32)
input = torch.reshape(input,(-1,1,5,5))
print(input.shape)
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, input):
        output = self.maxpool1(input)
        return output

maxpool =Tudui()
output = maxpool(input)
print(output)
step = 0
dataloader = DataLoader(dataset,batch_size=72)
for data in dataloader:
    imgs,targets = data
    output = maxpool(imgs)
    writer.add_images('input images',imgs,step)
    writer.add_images('output imgs',output,step)
    print('*********************************************')
    print(imgs.shape)
    print(output.shape)
    print('*********************************************')
    step = step + 1
writer.close()