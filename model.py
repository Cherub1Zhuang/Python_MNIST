from torch import nn
import torch.nn.functional as F

class Digit(nn.Module):
    def __init__(self):                    #继承父类
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)   # 输入通道，输出通道，5*5 kernel
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.fc1 = nn.Linear(20*10*10, 500)    # 全连接层，输入通道， 输出通道
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):          # 前馈
        input_size = x.size(0)     # 得到batch_size
        x = self.conv1(x)          # 输入：batch*1*28*28, 输出：batch*10*24*24(28-5+1)
        x = F.relu(x)              # 使表达能力更强大激活函数, 输出batch*10*24*24
        x = F.max_pool2d(x, 2, 2)  # 最大池化层，输入batch*10*24*24，输出batch*10*12*12

        x = self.conv2(x)          # 输入batch*10*12*12，输出batch*20*10*10
        x = F.relu(x)

        x = x.view(input_size, -1) # 拉平， 自动计算维度，20*10*10= 2000

        x = self.fc1(x)            # 输入batch*2000,输出batch*500
        x = F.relu(x)

        x = self.fc2(x)            # 输入batch*500 输出batch*10

        output = F.log_softmax(x, dim=1)  # 计算分类后每个数字的概率值

        return output