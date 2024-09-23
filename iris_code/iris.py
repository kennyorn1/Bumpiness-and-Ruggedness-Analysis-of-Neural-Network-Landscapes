import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import datasets
import numpy as np
import pandas as pd

# 加载鸢尾花数据集
iris = datasets.load_iris()
X = iris["data"].astype(np.float32)  # X为(150,4)的array数组
y = iris["target"].astype(np.int64)  # y为标签0,1,2

# 将数据分为训练集和测试集
# 将150个样本分成：训练：120；测试：30
train_ratio = 0.8
index = np.random.permutation(X.shape[0])
train_index = index[: int(X.shape[0] * train_ratio)]
test_index = index[int(X.shape[0] * train_ratio) :]
X_train, y_train = X[train_index], y[train_index]
X_test, y_test = X[test_index], y[test_index]


# 定义神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        torch.manual_seed(2)
        self.fc1 = nn.Linear(4, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 3)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = self.fc3(x)
        return x


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        torch.manual_seed(2)
        self.fc1 = nn.Linear(4, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 3)
        self.shortcut = nn.Sequential()

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x) + self.shortcut(x))
        x = self.fc3(x)
        return x


class BNNet(nn.Module):
    def __init__(self):
        super(BNNet, self).__init__()
        torch.manual_seed(2)
        self.fc1 = nn.Linear(4, 32)
        self.fc2 = nn.Linear(32, 32)
        self.norm = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 3)
        self.shortcut = nn.Sequential()

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = F.gelu(self.norm(x))
        x = self.fc3(x)
        return x


# 初始化模型、损失函数和优化器
criterion = nn.CrossEntropyLoss()
results = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 训练模型
num_epochs = 100
optList = ["with BN", "with res", "without res and BN"]
for opt in optList:
    if opt == "with res":
        model = ResNet().to(device)
    elif opt == "with BN":
        model = BNNet().to(device)
    elif opt == "without res and BN":
        model = Net().to(device)
    dimen = sum(x.numel() for x in model.parameters())
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    for epoch in range(num_epochs):
        inputs = torch.from_numpy(X_train)
        labels = torch.from_numpy(y_train)
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward(retain_graph=True)
        grad1sts = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        hessian_matrix = []
        for grad1st in grad1sts:
            shape = grad1st.shape
            if grad1st.ndim == 2:
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        grad2sts = torch.autograd.grad(
                            grad1st[i][j],
                            model.parameters(),
                            create_graph=True,
                        )
                        hessian_row = []
                        for grad2st in grad2sts:
                            for grad in grad2st.reshape(-1):
                                hessian_row.append(grad.item())
                        hessian_matrix.append(hessian_row)
            elif grad1st.ndim == 1:
                for i in range(shape[0]):
                    grad2sts = torch.autograd.grad(
                        grad1st[i], model.parameters(), create_graph=True
                    )
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
        with torch.no_grad():
            inputs = torch.from_numpy(X_test)
            labels = torch.from_numpy(y_test)
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)  # 获得outputs每行最大值的索引
            accuracy = (
                (predictions == labels).float().mean()
            )  # (predictions == labels):输出值为bool的Tensor
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
        print(epoch, loss.item(), accuracy.item(), ruggedness, bumpiness, avgtrace, opt)

        optimizer.step()
        optimizer.zero_grad()
data = pd.DataFrame(results)
data.to_csv("iris.csv", mode="a", header=False, index=False)


# # 评估模型
# with torch.no_grad():
#     inputs = torch.from_numpy(X_test)
#     labels = torch.from_numpy(y_test)
#     outputs = model(inputs)
#     _, predictions = torch.max(outputs, 1)  # 获得outputs每行最大值的索引
#     accuracy = (
#         (predictions == labels).float().mean()
#     )  # (predictions == labels):输出值为bool的Tensor
#     print("Accuracy: %.2f %%" % (accuracy.item() * 100))
