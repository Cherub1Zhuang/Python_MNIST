﻿# Python_MNIST
使用框架 pytorch

数据集：MINIST

数据集第一次会自动下载

先运行main.py，执行完模型参数会保存在checkpoints文件夹
运行完main.py在运行detect.py进行演示，程序会打印预测的值，以及展示原图片。

BATCH_SIZE = 128      # 每批处理的数据，可以修改
EPOCHS = 100       #Epoch可以修改，越大训练时间越长
LR = 0.001  # 学习率，可以修改，不建议修改
checkpoints=torch.load("checkpoints/checkponints_100.pth")    # 这里改成你自己最后一次的保存的模型参数即可，最后一次一般是效果最好的，但不一定

