import torchvision
import torch
#方式一加载
model1 = torch.load('vgg16_methold.pth')
print(model1)
#方式2加载
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))