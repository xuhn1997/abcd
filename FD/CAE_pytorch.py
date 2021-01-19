import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import pandas as pd
import numpy as np
from torch.autograd import Variable
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
# 获取训练集
from config_parameter import FILENAME, LIST_POSITION

from FD.vae_model import CAE

df = pd.read_csv(FILENAME)

train_data = np.array(df)

fd_y = train_data[:, 22, np.newaxis]

print(fd_y[0:10])
train_data = (train_data - np.mean(train_data, axis=0)) / np.std(train_data, axis=0)  # 标准化
# 获取相应的数据集
train_data = train_data[:, LIST_POSITION]

print("标准化后的数据维度是： %s" % str(train_data.shape))

# 获取A, B, C, D
train_x = train_data[:, 0:3]
train_y = train_data[:, 3]

# 将数据封装到dataloader里面
train_x = torch.from_numpy(train_x).float()
train_y = torch.from_numpy(train_y).float()

data = TensorDataset(train_x, train_y)
train_data_loader = DataLoader(dataset=data, batch_size=128)


# 定义优化器
def get_optimizer(net):
    """
    :param net:
    :return:
    """
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    return optimizer


# 获取网络及其优化器
FD_NET = CAE()

FD_NET.cuda()

# 定义优化器损失函数
mse_loss = nn.MSELoss()


def loss_function(W, y, recons_x, h, lam):
    """
    :param W: (N_hidden x N),encoder输出层的权重
    :param y:网络的输入b, 3,这里用y
    :param recons_x:全部的网络输出b, 1
    :param h:batch_size x N_hidden，encoder的输出
    :param lam:雅克比系数
    :return:
    """
    mse = mse_loss(recons_x, y)

    dh = h * (1 - h)

    w_sum = torch.sum(W ** 2, dim=1)

    w_sum = w_sum.unsqueeze(1)
    contractive_loss = torch.sum(torch.mm(dh ** 2, w_sum), 0)
    return mse + contractive_loss.mul_(lam)


optim = get_optimizer(FD_NET)

print("开始进行训练........")

FD_EPOCHS = 1000
lam = 1e-4
for epoch in range(FD_EPOCHS):
    # losses = []
    for step, (x, y) in enumerate(train_data_loader):
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        x = x.cuda()
        y = y.cuda()

        x = x.reshape(-1, 3)
        # print(x.shape)
        y = y.reshape(-1, 1)

        # 进入网络
        en_out, de_out = FD_NET(x)

        W = FD_NET.state_dict()['e_out.weight']

        loss = loss_function(W, y, de_out, en_out, lam)

        optim.zero_grad()
        loss.backward()
        optim.step()
        # losses.append(loss.item())
    # print(len(losses))
    if epoch % 1 == 0:
        print("epoch-----%s, losss is %s" % (str(epoch), str(loss.item())))

load_FD_model = "../fd_model_save/CAE_multiply_FD_3.pkl"
torch.save(FD_NET, load_FD_model)
