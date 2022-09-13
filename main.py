#1加载必要的库
import torch
import torch.optim as optim
from torch import nn
from torchvision import datasets, transforms

from model import Digit

#定义超参数(参数：由模型学习来决定的)数据太多一次放不完，切割
BATCH_SIZE = 128      # 每批处理的数据
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")      # CPU还是GPU？
EPOCHS = 100
LR = 0.001  # 学习率
 
 
#构建transform， 对图像进行各种处理(旋转拉伸，放大缩小等)
tranform = transforms.Compose([
    transforms.ToTensor(),       # 将图片转换成Tensor
    transforms.Normalize((0.1307,), (0.3081,))      # 均值和方差，正则化(对抗过拟合)：降低模型复杂度
])
 
 
#下载、加载数据集
from torch.utils.data import DataLoader
train_data = datasets.MNIST(root="./MNIST",
                            train=True,
                            transform=tranform,
                            download=True)
 
test_data = datasets.MNIST(root="./MNIST",
                           train=False,
                           transform=tranform,
                           download=True)
#加载数据集
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
 
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)


model = Digit().to(DEVICE)    # 创建模型部署到DEVICE
#定义优化器
optimizer = optim.Adam(model.parameters(),lr=LR)
loss_func = nn.CrossEntropyLoss()  # 目标标签是one-hotted
 

#train and test
for epoch in range(EPOCHS):
    model.train()
    for batch_id,(data,label) in enumerate(train_loader):
        data=data.to(DEVICE)
        label=label.to(DEVICE)
        output=model(data)
        loss=loss_func(output,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("Train Epoch : {}/{} \t Loss : {:.6f}".format(batch_id,epoch+1, loss.item()))
    if((epoch+1)%20==0):
        model.eval()
        for batch_id,(data,label) in enumerate(test_loader):
            data=data.to(DEVICE)
            label=label.to(DEVICE)
            output=model(data)
            loss=loss_func(output,label)
            print("Test Epoch : {}/{} \t Loss : {:.6f}".format(batch_id,epoch+1, loss.item()))
        torch.save(model.state_dict(),"checkpoints/checkponints_{}.pth".format(epoch+1))