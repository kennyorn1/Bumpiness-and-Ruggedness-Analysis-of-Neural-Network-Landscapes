# import tensorwatch as tw
# import torch
# import torchvision as tv
# import torchvision.transforms as transforms
# import torch.nn as nn
# import torch.optim as optim
# import argparse
# import numpy as np
# import pandas as pd
# import torch.nn.functional as F


# # 定义是否使用GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # 定义网络结构
# class LeNet(nn.Module):
#     def __init__(self):
#         super(LeNet, self).__init__()
#         self.conv1 = nn.Sequential(  # input_size=(1*28*28)
#             nn.Conv2d(1, 1, 5),  # padding=2保证输入输出尺寸相同
#             nn.GELU(),  # input_size=(6*28*28)
#         )
#         self.fc1 = nn.Sequential(nn.Linear(576, 10))

#     # 定义前向传播过程，输入为x
#     def forward(self, x):
#         x = self.conv1(x)
#         x = x.view(x.size()[0], -1)
#         x = self.fc1(x)
#         return x


# class ResLeNet(nn.Module):
#     def __init__(self):
#         super(ResLeNet, self).__init__()
#         self.conv1 = nn.Sequential(  # input_size=(1*28*28)
#             nn.Conv2d(1, 1, 5),  # padding=2保证输入输出尺寸相同
#             nn.GELU(),  # input_size=(6*28*28)
#         )
#         self.conv2 = nn.Sequential(  # input_size=(1*28*28)
#             nn.Conv2d(1, 1, 5, padding=2),  # padding=2保证输入输出尺寸相同
#         )
#         self.fc1 = nn.Sequential(nn.Linear(576, 10))

#     # 定义前向传播过程，输入为x
#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.gelu(x + self.conv2(x))
#         x = x.view(x.size()[0], -1)
#         x = self.fc1(x)
#         return x


# class BNLeNet(nn.Module):
#     def __init__(self):
#         super(BNLeNet, self).__init__()
#         self.conv1 = nn.Sequential(  # input_size=(1*28*28)
#             nn.Conv2d(1, 1, 5),  # padding=2保证输入输出尺寸相同
#             nn.BatchNorm2d(1),
#             nn.GELU(),  # input_size=(6*28*28)
#         )
#         self.fc1 = nn.Sequential(nn.Linear(576, 10))

#     # 定义前向传播过程，输入为x
#     def forward(self, x):
#         x = self.conv1(x)
#         x = x.view(x.size()[0], -1)
#         x = self.fc1(x)
#         return x


# # 其实就两句话

# model = ResLeNet()
# tw.draw_model(model, [1, 1, 28, 28])
# import tensorwatch as tw
# import torchvision.models

# alexnet_model = torchvision.models.alexnet()
# tw.draw_model(alexnet_model, [1, 3, 224, 224])
import torch
from torchviz import make_dot
from torchvision.models import vgg16  # 以 vgg16 为例

x = torch.randn(4, 3, 32, 32)  # 随机生成一个张量
model = vgg16()  # 实例化 vgg16，网络可以改成自己的网络
out = model(x)  # 将 x 输入网络
g = make_dot(out)  # 实例化 make_dot
g.view()  # 直接在当前路径下保存 pdf 并打开
# g.render(filename='netStructure/myNetModel', view=False, format='pdf')  # 保存 pdf 到指定路径不打开
