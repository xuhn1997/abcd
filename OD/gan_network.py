import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# 生成器网络
from config_parameter import GAN_INPUT_SIZE

"""
现在编写生成器的网络
"""


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(100, 128),
            nn.ReLU(True),
            #
            nn.Linear(128, 128),
            nn.ReLU(True),

            nn.Linear(128, 128),
            nn.ReLU(True),

            nn.Linear(128, GAN_INPUT_SIZE)
        )

    def forward(self, x):
        return self.net(x)


"""
现在是判别器网咯
"""


class Discriminator(nn.Module):
    def __init__(self, error_dim):
        super(Discriminator, self).__init__()
        """
        创建假和真的记录经过两个网络出来的
        """
        self.error_dim = error_dim
        # self.fc = nn.Sequential(
        #     nn.Linear(GAN_INPUT_SIZE, 16),
        #     nn.LeakyReLU(0.2),
        #
        #     # nn.Linear(64, 64),
        #     # nn.LeakyReLU(0.2)
        # )
        """
        然后与真实的或者fake的方差进行concat进入一个全连接层
        """
        self.out = nn.Sequential(
            nn.Linear(GAN_INPUT_SIZE + error_dim, 64),
            nn.LeakyReLU(0.2),

            nn.Linear(64, 64),
            nn.LeakyReLU(0.2),

            nn.Linear(64, 64),
            nn.LeakyReLU(0.2),

            nn.Linear(64, 1)
        )

    def forward(self, x_input, error):
        # 首先经过两层全连接
        # x = self.fc(x_input)

        # 出来的再和其误差进行concat
        x = torch.cat([x_input, error], dim=1)

        # 然后再进入全连接层
        return self.out(x)

# D = Discriminator(4)
#
# data = torch.randn(128, 37)
# error = torch.randn(128, 4)
# oo = D(data, error)
# print(oo.shape)
