import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from torchvision.datasets import CIFAR10
import flwr as fl#使用Flower实现联邦学习
from collections import OrderedDict
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
#========================================原始代码========================================
class Net(nn.Module):  
    def __init__(self):
        super(Net, self).__init__()  
        self.conv1 = nn.Conv2d(3, 6, 5)  
        self.pool = nn.MaxPool2d(2, 2)  
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120) 
        self.fc2 = nn.Linear(120, 84)  
        self.fc3 = nn.Linear(84, 10)  

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) 
        x = self.pool(F.relu(self.conv2(x))) 
        x = x.view(-1, 16 * 5 * 5)  
        x = F.relu(self.fc1(x))  
        x = F.relu(self.fc2(x))  
        return self.fc3(x)  

def train(net, trainloader, epochs):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):
        for images, labels in trainloader:
            optimizer.zero_grad()
            criterion(net(images.to(DEVICE)), labels.to(DEVICE)).backward()
            optimizer.step()

def test(net, testloader):  
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in testloader:
            outputs = net(images.to(DEVICE))
            loss += criterion(outputs, labels.to(DEVICE)).item()
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    return loss / len(testloader.dataset), correct / total

def load_data():  
    trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  
    trainset = CIFAR10("./data", train=True, download=True, transform=trf)  
    testset = CIFAR10("./data", train=False, download=True, transform=trf)  
    return DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(testset)

def load_model():
    return Net().to(DEVICE)  
#========================================原始代码========================================
def set_parameters(model, parameters):#使用从服务器接收的模型参数，用它们初始化模型。
    params_dict = zip(model.state_dict().keys(), parameters)#获取模型所有层的参数名列表作为键，与从服务器接收的模型参数关联起来作为参数字典
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})#将参数字典中的键值对转换为torch.tensor类型。
    model.load_state_dict(state_dict, strict=True)#更新模型的权重，如果缺少参数就报错，确保了模型始终有相同的尺寸和层

net = load_model()#定义模型
trainloader, testloader = load_data()#加载训练集和测试集
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):#从本地模型中提取参数，并将它们转换成一组NumPy数组
        return [val.cpu().numpy() for _, val in net.state_dict().items()]#模型参数对应的值传递给cpu转换为numpy数组
    def fit(self, parameters, config):#训练函数
        set_parameters(net,parameters)#将参数设置为从服务器上接收的参数
        train(net, trainloader, epochs=1)#训练
        return self.get_parameters({}), len(trainloader.dataset), {}#返回空配置、训练集大小、空字典（不在训练中计算指标）
    def evaluate(self, parameters, config):#评估全局模型在客户端的本地验证集上的性能
        set_parameters(net,parameters)#将参数设置为从服务器上接收的参数
        loss, accuracy = test(net, testloader)#用测试集评估损失和准确度
        return float(loss), len(testloader.dataset), {"accuracy": accuracy}#返回损失、测试集大小、准确度

fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient(),
)
# 联邦学习的客户端