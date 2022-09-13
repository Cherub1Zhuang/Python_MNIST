import PIL.Image as Image
import cv2
import torch
import torch.optim as optim
import torchvision
from matplotlib import pyplot as plt
from torch import nn
from torchvision import datasets, transforms
import numpy as np

from model import Digit

model=Digit()

checkpoints=torch.load("checkpoints/checkponints_100.pth")
model.load_state_dict(checkpoints)

tranform = transforms.Compose([
    transforms.ToTensor(),       # 将图片转换成Tensor
    transforms.Normalize((0.1307,), (0.3081,))      # 均值和方差，正则化(对抗过拟合)：降低模型复杂度
])
test_data = datasets.MNIST(root="./MNIST",
                           train=False,
                           transform=tranform,
                           download=True)
data=test_data[111][0]#要验证的图 111可以随便修改，要在test数量范围内
img = torchvision.utils.make_grid(data).numpy()
plt.imshow(np.transpose(img,(1,2,0)))
plt.show()

print(torch.max(model(data), 1)[1].data.numpy())

