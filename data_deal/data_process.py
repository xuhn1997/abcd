import pandas as pd
import numpy as np
import sklearn.preprocessing as sp
from sklearn.preprocessing import StandardScaler
# import tensorflow as tf
from config_parameter import OD_BATCH_SIZE, OD_TIME_STEP, LIST_POSITION


# 对于训练集进行处理

def get_transform_data(filename):
    """
    获取数据的标准化数据
    :param filename:
    :return:
    """
    Sc_data = StandardScaler()
    train_data = pd.read_csv(filename, header=None)
    train_data = np.array(train_data[0:4100])

    train_data_transformer = Sc_data.fit_transform(train_data)

    return Sc_data


class DataProcess:

    @staticmethod
    def data_load_multiply(filename):
        """
        多属性时数据的导入
        :param filename:
        :return:
        """
        """
        不用标准化包。通过计算其方差以及均值进行压缩
        """

        train_data1 = pd.read_csv(filename)
        train_data = np.array(train_data1)
        """开始压缩"""
        """
        不用标准化包。通过计算其方差以及均值进行压缩
        """
        train_data = (train_data - np.mean(train_data, axis=0)) / np.std(train_data, axis=0)  # 标准化

        # 取得相应的数据集
        train_data_x_Y = train_data[:, LIST_POSITION]

        x_shape = train_data_x_Y.shape[1] - 1
        # 最后获取得到测试集数据
        test_x = train_data_x_Y[:, 0:x_shape]

        return train_data_x_Y, test_x, x_shape

    # 获取训练集使得有往下移一步的效果
    @staticmethod
    def get_train_data_multiply(g_data, input_size1):  # G_data为原始的训练集数据
        """
        获取多属性时的训练集
        :param g_data:
        :param input_size1: OD左边的维度
        :return:返回的参数：train_x(None, time_step, input_size1), train_y(None, time_step, 1)
                            index:记录batch_size的index的一个数组，可以不用理会(使用keras的话)
        """
        batch_index = []
        train_begin = 0
        train_end = len(g_data)
        data_train = g_data[train_begin:train_end, :]
        train_x, train_y = [], []  # 训练集
        for i in range(len(g_data) - OD_TIME_STEP):
            if i % OD_BATCH_SIZE == 0:
                batch_index.append(i)
            x = data_train[i:i + OD_TIME_STEP, 0:input_size1]  # 取出左边的所有属性
            y = data_train[i:i + OD_TIME_STEP, input_size1, np.newaxis]  # 取出最后一个维度作为y
            train_x.append(x.tolist())
            train_y.append(y.tolist())
        batch_index.append((len(g_data) - OD_TIME_STEP))
        train_x = np.array(train_x)
        train_y = np.array(train_y)
        return batch_index, train_x, train_y

    @staticmethod
    def data_load_single(filename):
        """
        单属性时数据导入
        :param filename: 文件名
        :return: 返回原始数据的训练集以及测试集
        """

        # Sc_data = StandardScaler()

        train_data1 = pd.read_csv(filename)

        train_data = np.array(train_data1)  # 这里是取数据的过程，是取的单属性的数据！！！！！！

        # 要对所有的值进行压缩
        # train_data = Sc_data.fit_transform(train_data)
        train_data = (train_data - np.mean(train_data, axis=0)) / np.std(train_data, axis=0)  # 标准化

        # 压缩完之后取得相应的数值
        train_data = train_data[:, LIST_POSITION]

        # 获取测试集的过程
        test_X = train_data[:, 0, np.newaxis]
        x_shape = 1

        return train_data, test_X, x_shape

    # 获取训练集使得有往下移一步的效果
    @staticmethod
    def get_train_data_single(g_data):  # G_data为原始的训练集数据
        """
        获取单属性的处理后的数据
        :param g_data:
        :return: 返回的参数：train_x(None, time_step, input_size1), train_y(None, time_step, 1)
                            index:记录batch_size的index的一个数组，可以不用理会(使用keras的话)
        """
        batch_index = []
        train_begin = 0
        train_end = len(g_data)
        data_train = g_data[train_begin:train_end, :]
        train_x, train_y = [], []  # 训练集
        for i in range(len(g_data) - OD_TIME_STEP):
            if i % OD_BATCH_SIZE == 0:
                batch_index.append(i)
            x = data_train[i:i + OD_TIME_STEP, 0, np.newaxis]  # 取出左边的所有属性
            y = data_train[i:i + OD_TIME_STEP, 1, np.newaxis]  # 取出最后一个维度作为y
            train_x.append(x.tolist())
            train_y.append(y.tolist())
        batch_index.append((len(g_data) - OD_TIME_STEP))
        train_x = np.array(train_x)
        train_y = np.array(train_y)
        return batch_index, train_x, train_y
