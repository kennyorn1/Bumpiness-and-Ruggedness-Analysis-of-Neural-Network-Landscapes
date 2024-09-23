import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import pandas as pd
import torch.nn.functional as F


# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 定义网络结构
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(  # input_size=(1*28*28)
            nn.Conv2d(1, 1, 5),  # padding=2保证输入输出尺寸相同
            nn.GELU(),  # input_size=(6*28*28)
        )
        self.fc1 = nn.Sequential(nn.Linear(576, 10))

    # 定义前向传播过程，输入为x
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        return x


class ResLeNet(nn.Module):
    def __init__(self):
        super(ResLeNet, self).__init__()
        self.conv1 = nn.Sequential(  # input_size=(1*28*28)
            nn.Conv2d(1, 1, 5),  # padding=2保证输入输出尺寸相同
            nn.GELU(),  # input_size=(6*28*28)
        )
        self.conv2 = nn.Sequential(  # input_size=(1*28*28)
            nn.Conv2d(1, 1, 5, padding=2),  # padding=2保证输入输出尺寸相同
        )
        self.fc1 = nn.Sequential(nn.Linear(576, 10))

    # 定义前向传播过程，输入为x
    def forward(self, x):
        x = self.conv1(x)
        x = F.gelu(x + self.conv2(x))
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        return x


class BNLeNet(nn.Module):
    def __init__(self):
        super(BNLeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 1, 5),
            nn.BatchNorm2d(1),
            nn.GELU(),
        )
        self.fc1 = nn.Sequential(nn.Linear(576, 10))

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        return x


# 超参数设置
EPOCH = 10  # 遍历数据集次数
BATCH_SIZE = 60000  # 批处理尺寸(batch_size)
LR = 0.001  # 学习率

# 定义数据预处理方式
transform = transforms.ToTensor()

# 定义训练数据集
trainset = tv.datasets.MNIST(
    root="./data/", train=True, download=True, transform=transform
)
# 定义训练批处理数据
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)
# 定义测试数据集
testset = tv.datasets.MNIST(
    root="./data/", train=False, download=True, transform=transform
)

# 定义测试批处理数据
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数，通常用于多分类问题上


# 训练
if __name__ == "__main__":
    results = []
    optList = ["with BN", "with res", "without res and BN"]
    for opt in optList:
        if opt == "with res":
            net = ResLeNet().to(device)
        elif opt == "with BN":
            net = BNLeNet().to(device)
        elif opt == "without res and BN":
            net = LeNet().to(device)
        dimen = sum(x.numel() for x in net.parameters())
        optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)

        for epoch in range(EPOCH):
            sum_loss = 0.0
            # 数据读取
            for i, data in enumerate(trainloader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward(retain_graph=True)
                grad1sts = torch.autograd.grad(
                    loss, net.parameters(), create_graph=True
                )
                hessian_matrix = []
                for grad1st in grad1sts:
                    shape = grad1st.shape
                    if grad1st.ndim == 2:
                        print("calc2")
                        for i in range(shape[0]):
                            for j in range(shape[1]):
                                grad2sts = torch.autograd.grad(
                                    grad1st[i][j],
                                    net.parameters(),
                                    create_graph=True,
                                )
                                print(grad2sts)
                                hessian_row = []
                                for grad2st in grad2sts:
                                    for grad in grad2st.reshape(-1):
                                        hessian_row.append(grad.item())
                                hessian_matrix.append(hessian_row)
                    elif grad1st.ndim == 1:
                        print("calc1")
                        for i in range(shape[0]):
                            grad2sts = torch.autograd.grad(
                                grad1st[i], net.parameters(), create_graph=True
                            )
                            print(grad2sts)
                            hessian_row = []
                            for grad2st in grad2sts:
                                for grad in grad2st.reshape(-1):
                                    hessian_row.append(grad.item())
                            hessian_matrix.append(hessian_row)
                    elif grad1st.ndim == 4:
                        print("calc4")
                        for i in range(shape[0]):
                            for j in range(shape[1]):
                                for i1 in range(shape[2]):
                                    for j1 in range(shape[3]):
                                        grad2sts = torch.autograd.grad(
                                            grad1st[i][j][i1][j1],
                                            net.parameters(),
                                            create_graph=True,
                                        )
                                        print(grad2sts)
                                        hessian_row = []
                                        for grad2st in grad2sts:
                                            for grad in grad2st.reshape(-1):
                                                hessian_row.append(grad.item())
                                        hessian_matrix.append(hessian_row)

                hessian_matrix = np.array(hessian_matrix)

                eigenvalue, featurevector = np.linalg.eig(hessian_matrix)

                ruggedness = np.sum(np.abs(eigenvalue)) / dimen
                avgtrace = np.trace(hessian_matrix) / dimen
                bumpiness = avgtrace / ruggedness
                # 每训练100个batch打印一次平均loss
                print("[%d] loss: %.03f" % (epoch + 1, loss))
            # 每跑完一次epoch测试一下准确率
            with torch.no_grad():
                correct = 0
                total = 0
                for data in testloader:
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    outputs = net(images)
                    # 取得分最高的那个类
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum()
                accuracy = correct / total
                print("第%d个epoch的识别准确率为：%d%%" % (epoch + 1, (100 * correct / total)))
            results.append(
                [
                    epoch,
                    loss.item(),
                    accuracy.item(),
                    ruggedness,
                    bumpiness,
                    avgtrace,
                    opt,
                ]
            )
            print(
                epoch,
                loss.item(),
                accuracy.item(),
                ruggedness,
                bumpiness,
                avgtrace,
                opt,
            )

            optimizer.step()
            optimizer.zero_grad()
    data = pd.DataFrame(results)
    data.to_csv("minist.csv", mode="a", header=False, index=False)
