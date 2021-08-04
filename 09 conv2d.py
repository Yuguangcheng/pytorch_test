import torch
import torch.nn.functional as F

input = torch.tensor([[1,2,3,4,5],
                      [4,6,7,8,9],
                      [4,6,7,8,9],
                      [4,6,7,8,9],
                      [4,6,7,8,9]])#输入图像
kernal = torch.tensor([[1,2,3,4],
                      [4,6,7,8],
                      [4,6,7,8],
                      [4,6,7,8]])#卷积核
input = torch.reshape(input,(1,1,5,5))
kernal = torch.reshape(kernal,(1,1,4,4))
print(input.shape)
output1 = F.conv2d(input,kernal,stride=1)#步长为1
output2 = F.conv2d(input,kernal,stride=1,padding=1)#填充为1 
print(output1)
