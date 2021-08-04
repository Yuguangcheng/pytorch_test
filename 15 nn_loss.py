import torch
import torch.nn as nn
input = torch.tensor([1,2,3],dtype=torch.float32)
input = torch.flatten(input)
print(input)
target = torch.tensor([1,2,5],dtype=torch.float32)
target = torch.flatten(target)
l1loss = nn.L1Loss(reduction='mean')#绝对值求误差函数
output_l1loss = l1loss(input,target)
print(output_l1loss)
mseloss = nn.MSELoss()#平方求误差
output_mseloss = mseloss(input,target)
print(output_mseloss)
crossentroy = nn.CrossEntropyLoss()
x = torch.tensor([0.1,0.2,0.3])
x = torch.reshape(x, (1, -1))
print(x)
y = torch.tensor([1])
print(y)
output_crossentroy = crossentroy(x,y)
print(output_crossentroy)