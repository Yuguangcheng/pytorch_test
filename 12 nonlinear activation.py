#非线性激活
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR100('dataset_Crfar10',train=False,transform=torchvision.transforms.ToTensor(),download=True)
writer = SummaryWriter('logs')
input = torch.tensor([[-100,10],
                      [3,4]])
input = torch.reshape(input,(-1,1,2,2))
class ReLu(nn.Module):
    def __init__(self):
        super().__init__()
        self.ReLU=nn.Sigmoid()
    def forward(self,input):
        output = self.ReLU(input)
        return output
relu = ReLu()
output = relu(input)
print(output)
step = 0
dataloader = DataLoader(dataset,batch_size=72)
for data in dataloader:
    imgs,targets = data
    output = relu(imgs)
    writer.add_images('input images',imgs,step)
    writer.add_images('output imgs',output,step)
    print('*********************************************')
    print(imgs.shape)
    print(output.shape)
    print('*********************************************')
    step = step + 1
writer.close()