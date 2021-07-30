import torchvision

#创建transforms工具
from torch.utils.tensorboard import SummaryWriter

transforms_tool = torchvision.transforms.Compose([torchvision.transforms.ToTensor()]) #工具功能:转化为tensor数据类型
writer = SummaryWriter('logs')
#训练数据集
train_dataset = torchvision.datasets.CIFAR100('dataset_Crfar10',train=True,transform=transforms_tool,download=True)
#测试数据集
test_dataset = torchvision.datasets.CIFAR100('dataset_Crfar10',train=False,transform=transforms_tool,download=True)
print(train_dataset.classes)#查看该数据集的分类
print(train_dataset[0])
for i in range(10):
    img,target = train_dataset[i]
    writer.add_image('dataset',img,i)
writer.close()