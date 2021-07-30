import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('logs')
test_train  = torchvision.datasets.CIFAR100('D:\\anaconda_project\\dataset_Crfar10',train=False, transform=torchvision.transforms.ToTensor(),download=True)
data_loader = DataLoader(dataset=test_train,batch_size=64,shuffle=True,num_workers=0,drop_last=True)
step = 0
for data in data_loader:
    img,target = data
    writer.add_images('data_loader',img,step)
    step = step + 1
writer.close()