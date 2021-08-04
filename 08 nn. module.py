import torch
from torch import nn


class Add(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,input):
        output = input + 1
        return output
ad = Add()
x = torch.tensor(1.0)
output = ad(x)
print(output)