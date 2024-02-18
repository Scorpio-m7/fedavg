import matplotlib.pyplot as plt
import flwr as fl#使用Flower实现联邦学习
def weighted_average(metrics):#定义度量聚合函数
    accuracies = [num_examples * m["accuracy"] for num_examples,m in metrics]#将每个客户端的准确性乘以使用的示例数
    examples=[num_examples for num_examples, _ in metrics]
    return {"accuracy":sum(accuracies) / sum(examples)}#聚合和返回加权平均值
history=fl.server.start_server(#启动Flower服务器
    server_address="0.0.0.0:8080",#通信地址
    config=fl.server.ServerConfig(num_rounds=10),#设置循环10轮
    strategy=fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average,),#使用FedAvg
)
print(f"{history.metrics_distributed = }")#使用返回的History对象将结果进行可视化
global_accuracy_centralised = history.metrics_distributed['accuracy']
round = [data[0] for data in global_accuracy_centralised]#获取轮数
acc = [100.0*data[1] for data in global_accuracy_centralised]#获取准确率
plt.plot(round, acc)
plt.grid()
plt.ylabel('Accuracy (%)')
plt.xlabel('Round')
plt.title('CIFAR10 - Accuracy per round')
plt.show()
#联邦学习准确率达到60%
#联邦学习服务端
#服务端从其中一个客户端获取初始化模型参数，把初始化参数也发送给另外一个客户端，客户端本地拟合返回参数，服务器聚合参数发给两个客户端