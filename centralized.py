import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from torchvision.datasets import CIFAR10#使用CIFAR10数据集
#CIFAR-10数据集由10个类别的60000张32 x32彩色图像组成，每个类别6000张图像。有50000张训练图像和10000张测试图像。
#该数据集分为五个训练批次和一个测试批次，每个批次有10000张图像。测试批次包含从每个类别中随机选择的1000张图像。 
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#如果没有gpu使用cpu

class Net(nn.Module):#定义网络模型架构
    def __init__(self):#适用CIFAR10图像分类任务的典型CNN，两个卷积层和三个全连接层
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)#创建一个卷积层，输入通道数为3，输出通道数为6，卷积核大小为5x5。
        self.pool = nn.MaxPool2d(2, 2)#创建一个最大池化层，池化窗口大小为2x2。
        self.conv2 = nn.Conv2d(6, 16, 5)#创建另一个卷积层，输入通道数为6，输出通道数为16，卷积核大小为5x5。
        self.fc1 = nn.Linear(16 * 5 * 5, 120)#创建一个全连接层，输入大小为16x5x5，输出大小为120。
        self.fc2 = nn.Linear(120, 84)#创建另一个全连接层，输入大小为120，输出大小为84。
        self.fc3 = nn.Linear(84, 10)#创建最后一个全连接层，输入大小为84，输出大小为10。

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))#将输入x通过卷积层self.conv1，然后通过ReLU激活函数，再通过池化层self.pool进行处理。
        x = self.pool(F.relu(self.conv2(x)))#将处理后的结果再次通过卷积层self.conv2，然后通过ReLU激活函数，再通过池化层self.pool进行处理。
        x = x.view(-1, 16 * 5 * 5)#将处理后的结果展平为一个向量
        x = F.relu(self.fc1(x))#然后通过全连接层self.fc1，再通过ReLU激活函数。
        x = F.relu(self.fc2(x))#再次通过全连接层self.fc2，再通过ReLU激活函数。
        return self.fc3(x)#最后通过全连接层self.fc3

def train(net, trainloader, epochs):#根据训练集和训练次数训练网络
    criterion = torch.nn.CrossEntropyLoss()#创建交叉熵损失函数
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)#随机梯度下降，学习率0.001，动量为0.9
    for _ in range(epochs):
        for images,labels in trainloader:
            optimizer.zero_grad()#梯度清零
            criterion(net(images.to(DEVICE)), labels.to(DEVICE)).backward()#将图像数据送入模型并转换至设备，计算模型输出与真实标签之间的交叉熵损失。然后反向传播计算参数梯度。
            optimizer.step()#梯度更新
def test(net, testloader):#评估函数，并计算损失和准确率
    criterion = torch.nn.CrossEntropyLoss()#创建交叉熵损失函数
    correct,total, loss = 0, 0,0.0#初始化正确分类的数量、总样本数量、损失值
    with torch.no_grad():#禁用梯度计算
        for images,labels in testloader:
            outputs=net(images.to(DEVICE))#图像传给模型
            loss += criterion(outputs, labels.to(DEVICE)).item()#累计模型损失
            total+=labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()#累加正确数量
    return loss/len(testloader.dataset),correct/total#返回损失和准确度

def load_data():#加载测试集和训练集的数据加载器
    trf=Compose([ToTensor(),Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])#将图像转换为张量并应用归一化的变换
    trainset=CIFAR10("./data", train=True, download=True, transform=trf)#准备训练集
    testset=CIFAR10("./data", train=False, download=True, transform=trf)#准备测试集
    #================================以下代码是展示数据所用================================
    print(trainset)#快速预览训练集,5万个训练样本
    print(testset)#快速预览测试集,1万个测试样本
    classes = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")#图片有十个分类
    (data, label) = trainset[100]#船的图片
    print(classes[label], "\t", data.shape)#查看第100个样本的标签
    plt.imshow((data.permute(1, 2, 0) + 1) / 2)#查看第100个样本的图像
    plt.show()
    #从数据集中可视化32张图像
    fig, axs = plt.subplots(4, 8, figsize=(15, 8))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    axs = axs.ravel()
    for i in range(32):
        data, label = trainset[i]
        data = (data.permute(1, 2, 0) + 1) / 2#数字标签对应类别
        axs[i].imshow(data)
        axs[i].set_title(classes[label])
    plt.show()
    #================================数据展示结束================================
    return DataLoader(trainset,batch_size=32,shuffle=True), DataLoader(testset)

def load_model():
    return Net().to(DEVICE)#返回模型并转换到正确的设备

if __name__ == "__main__":
    net=load_model()
    num_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"{num_parameters = }")#模型参数的数量为62006
    trainloader, testloader = load_data()
    train(net, trainloader, 5)#训练网络5次
    loss,accuracy=test(net, testloader)#获得损失和准确率
    print("Test loss: {:.5f}, Test accuracy: {:.5f}".format(loss, accuracy))
    
#采用集中式训练：传统的机器学习方式训练（没有启动联邦学习时）的准确率为0.5左右