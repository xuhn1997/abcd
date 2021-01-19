import torch

from config_parameter import OD_RNN_UNIT, OD_TIME_STEP
from data_deal.data_process import *
from torch import nn

import torch.nn.functional as F

rnn_unit = OD_RNN_UNIT


class RNN_SINGLE(nn.Module):
    """
       其中必须传的参数是input_size1,为OD左属性的个数
    """

    def __init__(self, input_size1):
        super(RNN_SINGLE, self).__init__()
        self.input_size1 = input_size1
        # 开始定义模型的层数
        self.rnn = nn.LSTM(
            input_size=input_size1,
            hidden_size=rnn_unit,
            num_layers=1,
            batch_first=True,
        )
        # 加入一个输出层
        self.out = nn.Linear(rnn_unit, 1)

    def forward(self, x_input):
        lstm_out, _ = self.rnn(x_input, None)
        lstm_out = lstm_out.contiguous().view(-1, rnn_unit)

        prediction = self.out(lstm_out)

        return prediction


# model = RNN_SINGLE(input_size1=1)
# print(model)


class RNN_MULTIPLY(nn.Module):
    """
       其中必须传的参数是input_size1,为OD左属性的个数
    """

    def __init__(self, input_size1):
        super(RNN_MULTIPLY, self).__init__()
        self.input_size1 = input_size1
        # 开始定义模型的层数
        self.input = nn.Linear(self.input_size1, 1)
        self.rnn = nn.LSTM(
            input_size=1,
            hidden_size=rnn_unit,
            num_layers=1,
            batch_first=True
        )
        # 加入一个输出层
        self.out = nn.Linear(rnn_unit, 1)

    def forward(self, x_input):
        x_input = x_input.contiguous().view(-1, self.input_size1)
        input_temp = self.input(x_input)

        input_temp = input_temp.reshape(-1, OD_TIME_STEP, 1)

        lstm_out, _ = self.rnn(input_temp, None)

        lstm_out = lstm_out.reshape(-1, rnn_unit)

        prediction = self.out(lstm_out)

        return prediction


#
# model_multiply = RNN_MULTIPLY(input_size1=3)
# print(model_multiply)

# # 定义CNN的网络
# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#
#         self.input = nn.Sequential(
#             nn.Linear(3, 8),
#             nn.LeakyReLU()
#         )
#         self.conv1 = nn.Sequential(
#             nn.Conv1d(in_channels=1,
#                       out_channels=8,
#                       kernel_size=3),
#             nn.MaxPool1d(kernel_size=2),
#             # nn.MaxPool1d(kernel_size=2, padding=1),
#             nn.LeakyReLU()
#         )
#         self.hidden = nn.Sequential(
#             nn.Linear(3, 8),
#             nn.LeakyReLU()
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv1d(in_channels=8,
#                       out_channels=16,
#                       kernel_size=3),
#             nn.MaxPool1d(kernel_size=2),
#             # nn.MaxPool1d(kernel_size=2, padding=1),
#             nn.LeakyReLU()
#         )
#         self.fc = nn.Linear(48, 1)
#
#     def forward(self, x):
#         x = self.input(x)
#
#         x = self.conv1(x)
#
#         x = self.hidden(x)
#
#         x = self.conv2(x)
#
#         x = x.view(x.size(0), -1)
#
#         out = self.fc(x)
#         return out
#

# cnn = CNN()
# print(cnn)
# data = torch.rand(128, 1, 3)
# out = cnn(data)
# print(out.shape)
# print(cnn)
