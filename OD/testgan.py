import warnings

warnings.filterwarnings("ignore")

import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import torch
from torch import nn
from OD.od_recover import several_od_recover, LEFT_ATTRIBUTES_POSITION, get_error_item
from config_parameter import RIGHT_ATTRIBUTES_POSITION, GAN_INPUT_SIZE, FILENAME

from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from OD.gan_network import Generator, Discriminator

# 设置超参数部分
NOISE_DIM = 100
batch_size = 128

# 数据集获取部分
# filename = "../data/b_deal.csv"
# train_data = pd.read_csv(filename, header=None)

train_data = pd.read_csv(FILENAME)

train_data = np.array(train_data)  # 转化成数组格式

print("**********")
print(train_data.shape)
x_input = train_data.shape[0]

"""试着对数据进行归一化"""
"""
不采用包自己进行标准化...
"""
train_data = (train_data - np.mean(train_data, axis=0)) / np.std(train_data, axis=0)  # 标准化

# 封装到dataloader里面
train_data = torch.from_numpy(train_data).float()
y = torch.Tensor(x_input, 2).uniform_(-1, 1)

data = TensorDataset(train_data, y)

train_data_loader = DataLoader(dataset=data, num_workers=8,
                               batch_size=batch_size)



# 定义好交叉熵损失函数
bce_loss = nn.BCEWithLogitsLoss()

# mse_loss = nn.L1Loss()  # 换成L1正则化损失


# 标准化数据的话换成平方差

mse_loss = nn.MSELoss()


def discriminator_loss(logits_real, logits_fake):
    """
    定义判别器损失函数
    :param real_data:
    :param beta:
    :param od_right_attribute_position:
    :param od_input:
    :param generator_out:
    :param logits_real:
    :param logits_fake:
    :return:
    """

    size = logits_real.shape[0]
    true_labels = torch.ones(size, 1).float().cuda()
    false_labels = torch.zeros(size, 1).float().cuda()

    loss = (bce_loss(logits_real, true_labels) + bce_loss(logits_fake, false_labels)) * 0.5

    return loss


def generate_loss(logits_fake, generator_out, od_right_attribute_position, od_input, beta):
    """
    生成器的损失函数
    :param real_data:
    :param beta:
    :param od_input:
    :param od_right_attribute_position:
    :param generator_out:
    :param logits_fake:
    :return:
    """

    size = logits_fake.shape[0]
    true_labels = torch.ones(size, 1).float().cuda()

    generator_out_data = None
    for i in od_right_attribute_position:
        tmp = generator_out[:, i].unsqueeze(1)

        if generator_out_data is None:
            generator_out_data = tmp
        else:
            generator_out_data = torch.cat([generator_out_data, tmp], dim=1)

    loss = bce_loss(logits_fake, true_labels) + beta * mse_loss(generator_out_data, od_input)
    return loss


# 定义优化器
def get_optimizer(net):
    """
    :param net:
    :return:
    """
    optimizer = torch.optim.Adam(net.parameters(), lr=2e-4)
    return optimizer


# 分别获取网络
# D = discriminator()
# G = generator()
D = Discriminator(len(RIGHT_ATTRIBUTES_POSITION))  # 要进行concat右属性的个数

G = Generator()

D.cuda()
G.cuda()

# 获取优化器
D_optim = get_optimizer(D)
G_optim = get_optimizer(G)


# def train_gan(D_net)
# 开始进行训练
def run():
    """
    :return:
    """
    for epoch in range(100000):
        for real_data, _ in train_data_loader:
            bs = real_data.shape[0]

            """
            判别器网络！！！
            """
            real_data = real_data.cuda()
            # 获取噪声数据
            sample_noise = torch.Tensor(bs, NOISE_DIM).uniform_(-1, 1)
            sample_noise = sample_noise.cuda()
            # 获取真实数据的判别器得分
            # 还有获得误差项
            error_item = get_error_item(real_data, od_right_attribute_position=RIGHT_ATTRIBUTES_POSITION,
                                        left_position=LEFT_ATTRIBUTES_POSITION)
            error_item = torch.from_numpy(error_item)
            error_item = torch.tensor(error_item, dtype=torch.float32).cuda()

            logits_real = D(real_data, error_item)  # 获得真实样本的标签

            # 生成假的数据集
            fake_data = G(sample_noise)
            # 获取假的数据的判别器得分
            # 还要获得误差项
            f_error_item = get_error_item(fake_data, od_right_attribute_position=RIGHT_ATTRIBUTES_POSITION,
                                          left_position=LEFT_ATTRIBUTES_POSITION)

            f_error_item = torch.from_numpy(f_error_item)
            f_error_item = torch.tensor(f_error_item, dtype=torch.float32).cuda()

            logits_fake = D(fake_data, f_error_item)  # 获得假的样本的标签

            # 获取loss函数
            d_total_error = discriminator_loss(logits_real, logits_fake)

            D_optim.zero_grad()
            d_total_error.backward()
            D_optim.step()  # 优化判别网络

            """
            生成器网络！！！！！！
            """
            fake_data_g = G(sample_noise)

            # 还要获得error的误差项
            fake_error_item_g = get_error_item(data=fake_data_g, od_right_attribute_position=RIGHT_ATTRIBUTES_POSITION,
                                               left_position=LEFT_ATTRIBUTES_POSITION)
            # 记得变成torch格式
            fake_error_item_g = torch.from_numpy(fake_error_item_g)
            fake_error_item_g = torch.tensor(fake_error_item_g, dtype=torch.float32).cuda()

            gen_logits_fake = D(fake_data_g, fake_error_item_g)

            # 要恢复成原来的数据, 以及ganout出来的数据
            od_right_attribute_g = several_od_recover(fake_data_g, LEFT_ATTRIBUTES_POSITION)

            # od_right_attribute_g = od_right_attribute_g.cpu().detach().numpy()
            # 转化成tensor形式 试着多属性注释的
            od_right_attribute_g = torch.from_numpy(od_right_attribute_g)
            od_right_attribute_g = torch.tensor(od_right_attribute_g, dtype=torch.float32).cuda()

            g_error = generate_loss(gen_logits_fake, fake_data_g, RIGHT_ATTRIBUTES_POSITION,
                                    od_right_attribute_g,
                                    beta=0.5)

            G_optim.zero_grad()
            g_error.backward(retain_graph=True)
            G_optim.step()
        if epoch % 1 == 0:
            print(f'epoch_g_error: {epoch:3} loss: {g_error.item():10.8f}')
            print(f'epoch_d_error: {epoch:3} loss: {d_total_error.item():10.8f}')
            print("************************************")
            print("************************************")

    # 保存网洛
    load_single_name = "../gan_model/ncv_all_model.pkl"
    torch.save(G, load_single_name)
    print("save successfully........")


if __name__ == '__main__':
    run()
