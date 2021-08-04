import torch
import torchvision
from torch import nn, optim
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10('dataset_Crfar10',train = False,transform=torchvision.transforms.ToTensor()
                                        ,download=True)
dataloader = DataLoader(dataset,batch_size=1)
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model1 = Sequential(
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

    def forward(self, x):
        x = self.model1(x)
        return x


loss = nn.CrossEntropyLoss()#设置误差函数
tudui = Tudui()
optimizer= optim.SGD(tudui.parameters(),lr=0.01)#设置优化器
for epoch in range(10):
    sum_loss = 0
    for data in dataloader:
        imgs, targets = data
        optimizer.zero_grad()#初始梯度设置为0
        outputs = tudui(imgs)
        result_loss = loss(outputs, targets)#计算输出和训练数据标间之间的误差
        result_loss.backward()#对误差进行反向传播
        optimizer.step()#更新权重
        sum_loss = sum_loss + result_loss
    print(sum_loss)
