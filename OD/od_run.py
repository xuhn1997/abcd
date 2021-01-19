import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from data_deal.data_process import *
from config_parameter import *
from OD.od_model import RNN_SINGLE, RNN_MULTIPLY
import torch
from torch import nn
import torch.utils.data as Data

# """
#    测试单属性的代码
# """

# train_data, test_data, input_left = DataProcess.data_load_single(FILENAME)
#
# # 将数据转化成三维的数据
# _, train_x, train_y = DataProcess.get_train_data_single(train_data)
# print(train_x.shape)
# print(train_y.shape)
#
# # 将数据转化torch的tensor形式
# train_x = torch.from_numpy(train_x)
# train_y = torch.from_numpy(train_y)
#
# rnn_single = RNN_SINGLE(input_size1=input_left)
# rnn_single.cuda()
#
# # 定义损失函数以及优化器
# loss_function = nn.MSELoss()
# # 损失函数再改成绝对值 不进行标准化之后
# # loss_function = nn.L1Loss()
# #
# optimizer = torch.optim.Adam(rnn_single.parameters(), lr=OD_LR)
#
# # 封装数据到dataloader
# # print(train_y.size(), train_x.size())
# deal_dataset = Data.TensorDataset(train_x, train_y)
# loader = Data.DataLoader(
#     dataset=deal_dataset,
#     batch_size=OD_BATCH_SIZE,
#     shuffle=False,
#     num_workers=1,
# )
#
#
# # 开始训练模型
# def run_single_model():
#     """
#     训练模型的函数
#     :return:
#     """
#     for epoch in range(OD_EPOCHS):
#         for step, (b_x, b_y) in enumerate(loader):
#             b_x = torch.tensor(b_x, dtype=torch.float32)
#             b_y = torch.tensor(b_y, dtype=torch.float32)
#             b_y = b_y.cuda()
#             b_x = b_x.cuda()
#             # 将b_y进行维度转化
#             b_y = b_y.contiguous().view(-1, 1)
#             output = rnn_single(b_x)
#             loss = loss_function(output, b_y)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#         if epoch % 2000 == 0:
#             print(f'epoch: {epoch:3} loss: {loss.item():10.8f}')
#
#     # 保存网络结构
#     load_single_name = "../order_model_single/order_model_single3.pkl"
#     torch.save(rnn_single, load_single_name)
#     print("save successfully......")


# """
#    测试多属性时使用的代码
# """
train_data, test_data, input_left = DataProcess.data_load_multiply(FILENAME)

# 将数据转化成三维的数据
_, train_x, train_y = DataProcess.get_train_data_multiply(train_data, input_left)

# 将数据转化torch的tensor形式
train_x = torch.from_numpy(train_x)
train_y = torch.from_numpy(train_y)

rnn_multiply = RNN_MULTIPLY(input_size1=input_left)
rnn_multiply.cuda()
# 定义损失函数以及优化器
loss_function = nn.MSELoss()
# loss_function = nn.L1Loss()
optimizer = torch.optim.Adam(rnn_multiply.parameters(), lr=OD_LR)

# 封装数据到dataloader
print(train_y.size(), train_x.size())
deal_dataset = Data.TensorDataset(train_x, train_y)
loader = Data.DataLoader(
    dataset=deal_dataset,
    batch_size=OD_BATCH_SIZE,
    shuffle=False,
    num_workers=1,
)


# 开始训练模型
def run_multiply_model():
    """
    训练模型的函数
    :return:
    """
    for epoch in range(OD_EPOCHS):
        for step, (b_x, b_y) in enumerate(loader):

            b_x = torch.tensor(b_x, dtype=torch.float32)
            b_y = torch.tensor(b_y, dtype=torch.float32)
            b_x = b_x.cuda()
            b_y = b_y.cuda()
            # 将b_y进行维度转化
            b_y = b_y.reshape(-1, 1)
            output = rnn_multiply(b_x)
            loss = loss_function(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 2000 == 0:
            print(f'epoch: {epoch:3} loss: {loss.item():10.8f}')
    # 打印每一层的参数名和参数值
    for layer in rnn_multiply.modules():
        if isinstance(layer, nn.Linear):
            weights = layer.weight
            print(layer.weight)
            print(weights.shape)
            break

    # 保存网络结构
    load_multiply_name = "../order_model_multiply/order_model_multiply4.pkl"
    torch.save(rnn_multiply, load_multiply_name)


#
if __name__ == '__main__':
    # run_single_model()
    run_multiply_model()
