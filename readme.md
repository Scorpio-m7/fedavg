通过flower实现了联邦学习中的fedavg，可以随时进行客户端的上下线并可以对客户端进行随机选择，在客户端上下线过程中训练不被中断，训练参数可以调控，训练过程可视化。
centralized.py实现了集中式训练作为对比

首先运行centralized.py会将数据集下载，然后会展示下载的数据集，训练完成后打印出准确度。
运行server客户端会在本地8080端口开启联邦学习的服务端，修改client.py中的IP为服务器地址后在客户端运行，客户端连接服务器后开始模型训练。