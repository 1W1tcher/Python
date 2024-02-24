import time
import torch
import torchvision
import torch.nn.init as init
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

#数据集加载
cifar10 = torchvision.datasets.CIFAR10(
root = r'D:\XZ\cifar-10-python',
train = True,
download = False
)
cifar10_test = torchvision.datasets.CIFAR10(
root = r'D:\XZ\cifar-10-python',
train = False,
download = False
)

print(cifar10)
print(cifar10_test)

#模型构建
class MLP(torch.nn.Module):
    def __init__(self, num_i, num_h, num_j, num_o):
        super(MLP, self).__init__()

        self.linear1 = torch.nn.Linear(num_i, num_h)  # 输入层到第一隐藏层的线性转换
        init.xavier_uniform_(self.linear1.weight)  # 使用Xavier初始化
        self.relu1 = torch.nn.ReLU()
        self.bn1 = torch.nn.BatchNorm1d(num_h)  # 添加批量归一化层
        self.linear2 = torch.nn.Linear(num_h, num_h)  # 第一隐藏层到第二隐藏层的线性转换
        self.relu2 = torch.nn.ReLU()
        self.bn2 = torch.nn.BatchNorm1d(num_h)
        self.linear3 = torch.nn.Linear(num_h, num_j)  # 第二隐藏层到第三隐藏层的线性转换
        self.relu3 = torch.nn.ReLU()
        self.bn3 = torch.nn.BatchNorm1d(num_j)
        self.linear4 = torch.nn.Linear(num_j, num_o)  # 第三隐藏层到输出层的线性转换
        torch.nn.init.xavier_uniform_(self.linear4.weight)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.relu3(x)
        x = self.linear4(x)
        return x


# 定义一个数据变换，将图像转换为张量
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为张量
])


# 创建数据加载器时应用数据变换
cifar10.transform = transform  # 将 transform 应用于数据集
data_loader_train = DataLoader(cifar10, batch_size=64, shuffle=True)
# 创建测试数据加载器
cifar10_test.transform = transform  # 将 transform 应用于数据集
data_loader_test = DataLoader(cifar10_test, batch_size=64, shuffle=False)



data_train = []  # 创建一个空列表，用于存储训练数据

for data in data_loader_train:  # 假设 data_loader_train 包含了训练数据
    inputs, labels = data
    # 将输入数据和标签添加到 data_train 列表中
    data_train.append((inputs, labels))

# 将 data_train 转化为 PyTorch 张量（如果它不是已经的话）
#data_train = torch.tensor(data_train)


#跟踪模型的预测和真实标签
train_predictions = []
train_labels = []
validation_predictions = []
validation_labels = []




#创建一个TensorBoard的SummaryWriter对象
writer = SummaryWriter('./path/to/log')

#模型训练
def train(model):
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    epochs = 30
    for epoch in range(epochs):
        sum_loss = 0
        train_correct = 0

        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        for data in data_loader_train:
            inputs, labels = data
            inputs = torch.flatten(inputs, start_dim=1)
            outputs = model(inputs)
            optimizer.zero_grad()
            loss = loss_func(outputs, labels)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)


            optimizer.step()
            _, id = torch.max(outputs.data, 1)
            sum_loss += loss.item()
            train_correct += torch.sum(id == labels).item()

            # 记录每个批次的预测和标签
            train_predictions.extend(id.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

            # 将损失和准确度记录到TensorBoard
            writer.add_scalar('Loss', sum_loss / len(data_loader_train), epoch)
            writer.add_scalar('Accuracy', 100 * train_correct / len(cifar10), epoch)

        print('[%d/%d] loss:%.3f, correct:%.3f%%, time:%s' %
              (epoch + 1, epochs, sum_loss / len(data_loader_train),
               100 * train_correct / len(cifar10),
               time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))
    model.eval()


# 创建MLP模型的实例并调用train方法
model = MLP(num_i=3072, num_h=512, num_j=256, num_o=10)
train(model)

#模型测试
def test(model, data_loader_test):
    test_correct = 0
    for data in data_loader_test:
        inputs, labels = data
        #inputs, labels = Variable(inputs).cpu(), Variable(labels).cpu()
        inputs = torch.flatten(inputs, start_dim = 1)
        outputs = model(inputs)
        _, id = torch.max(outputs.data, 1)
        test_correct += torch.sum(id == labels.data)

        # 记录每个批次的预测和标签
        validation_predictions.extend(id.cpu().numpy())
        validation_labels.extend(labels.cpu().numpy())

    print(f'Accuracy on test set : {100 * test_correct / len(cifar10_test):.3f}%')

 # 在训练完成后调用测试函数
test(model, data_loader_test)


# 计算训练集的混淆矩阵
train_cm = confusion_matrix(train_labels, train_predictions)

# 计算验证集的混淆矩阵
validation_cm = confusion_matrix(validation_labels, validation_predictions)

# 可视化混淆矩阵
plt.figure(figsize=(10, 8))

plt.subplot(2, 1, 1)
sns.heatmap(train_cm, annot=True, fmt="d", cmap="Blues")
plt.title("Train Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")

plt.subplot(2, 1, 2)
sns.heatmap(validation_cm, annot=True, fmt="d", cmap="Blues")
plt.title("Validation Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")

plt.tight_layout()
plt.show()
