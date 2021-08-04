import torch
import torchvision
import torch.nn as nn
vgg16 = torchvision.models.vgg16(pretrained=False)
#方式1
torch.save(vgg16,'vgg16_methold.pth')
#方式2 推荐
torch.save(vgg16.state_dict(),'vgg16_methold2.pth')