import torch
from data_deal.data_process import *
from config_parameter import *
import numpy as np

import sys
import os
import time

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


# 将多个od融合的代码
def several_od_recover(gan_out_data, left_position):
    """
    这个函数是多个od进行结合恢复的函数
    :param gan_out_data:生成器输出的数据表
    :param left_position:od的左属性的位置, 格式为一个字典序
    {'1': [0],
    '2': [8, 9, 10]}
    :return:
    """

    """gan出来的数据集不需要动"""
    # # 开始进行结合，要遍历该字典
    # 获取gan_out_data的数据，并去掉其梯度。。。。
    gan_out_data_numpy = gan_out_data.cpu().detach().numpy()

    pred_all = None
    for od_id, od_id_left_position in left_position.items():
        if len(od_id_left_position) == 1:
            if od_id == '0':
                """
                表明为FD
                """
                fd_left_data_single = None
                for i in od_id_left_position:
                    fd_left_data_single = gan_out_data_numpy[:, i]
                    fd_left_data_single = np.reshape(fd_left_data_single, (-1, 1))

                # 再封装回tensor里面
                fd_left_data_single = torch.from_numpy(fd_left_data_single)
                fd_left_data_single = torch.tensor(fd_left_data_single, dtype=torch.float32).cuda()

                load_fd_model = "../fd_model_save/fd_model.pkl"
                fd_net = torch.load(load_fd_model)
                fd_net.cuda()
                pred_fd = fd_net(fd_left_data_single)
                """
                 由于不符合transfomer使用的GPU全部是要CPU
                """
                pred_fd_numpy = pred_fd.cpu().detach().numpy()
                # 修改维度
                pred_fd_numpy = np.reshape(pred_fd_numpy, (-1, 1))

                if pred_all is None:
                    # pred_all = pred_fd
                    # 以上为numpy形式时
                    pred_all = pred_fd_numpy
                else:
                    # pred_all = torch.cat([pred_all, pred_fd], dim=1)  # 后面出来要去掉其梯度
                    pred_all = np.concatenate((pred_all, pred_fd_numpy), axis=1)

            else:
                """
                表示这个od为单属性
                """
                od_left_data_single = None
                for i in od_id_left_position:
                    od_left_data_single = gan_out_data_numpy[:, i]
                    od_left_data_single = np.reshape(od_left_data_single, (-1, 1))

                # 再封装回tensor里面
                od_left_data_single = torch.from_numpy(od_left_data_single)
                od_left_data_single = torch.tensor(od_left_data_single, dtype=torch.float32).cuda()

                # 获取完左边属性的数据之后进行获取网络结构进行融合
                od_left_data_single = od_left_data_single.view(-1, 20, 1)

                load_single_name = "../order_model_single/order_model_single" + str(od_id) + ".pkl"
                net1 = torch.load(load_single_name)
                net1.cuda()
                pred_single = net1(od_left_data_single)  # （-1， 1） 为tensor的形式

                """
                 由于不符合transfomer使用的GPU全部是要CPU
                """
                pred_single_numpy = pred_single.cpu().detach().numpy()
                # 修改维度
                pred_single_numpy = np.reshape(pred_single_numpy, (-1, 1))
                if pred_all is None:
                    # pred_all = pred_single
                    # 以上为numpy形式时
                    pred_all = pred_single_numpy
                else:
                    # pred_all = torch.cat([pred_all, pred_single], dim=1)  # 后面出来要去掉其梯度
                    # 以上出来恢复数据为numpy形式时
                    pred_all = np.concatenate((pred_all, pred_single_numpy), axis=1)
        else:
            """
            判断为多属性的FD时如何处理
            """
            if od_id == '5':
                fd_left_data_multiply = None
                for i in od_id_left_position:
                    tmp = gan_out_data_numpy[:, i]
                    tmp = np.reshape(tmp, (-1, 1))
                    if fd_left_data_multiply is None:
                        fd_left_data_multiply = tmp
                    else:
                        fd_left_data_multiply = np.concatenate((fd_left_data_multiply, tmp), axis=1)

                # 然后再变为tensor送回模型中去
                fd_left_data_multiply = torch.from_numpy(fd_left_data_multiply).float()
                # 变成三维进入transfomer
                fd_left_data_multiply = fd_left_data_multiply.unsqueeze(-1)

                # 加载多属性的网络
                load_fd_mul_model = "../fd_model_save/transformer_fd_model.pkl"

                mul_fd_net = torch.load(load_fd_mul_model)

                pred_fd_multiply = mul_fd_net(fd_left_data_multiply)

                """
                  由于不符合transfomer使用的GPU全部是要CPU
                """
                pred_fd_multiply_numpy = pred_fd_multiply.detach().numpy()
                # 修改维度
                pred_fd_multiply_numpy = np.reshape(pred_fd_multiply_numpy, (-1, 1))
                if pred_all is None:
                    # pred_all = pred_fd_multiply

                    pred_all = pred_fd_multiply_numpy
                else:
                    # 以上出来恢复数据为numpy形式时
                    pred_all = np.concatenate((pred_all, pred_fd_multiply_numpy), axis=1)

            else:
                """
                   则为多属性的od。处理如下
                """
                od_left_data_multiply = None
                for i in od_id_left_position:
                    tmp = gan_out_data_numpy[:, i]
                    tmp = np.reshape(tmp, (-1, 1))
                    if od_left_data_multiply is None:
                        od_left_data_multiply = tmp
                    else:
                        od_left_data_multiply = np.concatenate((od_left_data_multiply, tmp), axis=1)

                # 然后再变为tensor送回模型中去
                od_left_data_multiply = torch.from_numpy(od_left_data_multiply)
                od_left_data_multiply = torch.tensor(od_left_data_multiply, dtype=torch.float32).cuda()

                # 再进行转化成三维的
                od_left_data_multiply = od_left_data_multiply.view(-1, 20, len(od_id_left_position))  # 左边属性的维度

                # 获取多属性保存好的网络
                load_multiply_name = "../order_model_multiply/order_model_multiply" + str(od_id) + ".pkl"
                net2 = torch.load(load_multiply_name)
                net2.cuda()
                pred_multiply = net2(od_left_data_multiply)

                """
                由于不符合transfomer使用的GPU全部是要CPU
                """
                pred_multiply_numpy = pred_multiply.cpu().detach().numpy()
                # 修改维度
                pred_multiply_numpy = np.reshape(pred_multiply_numpy, (-1, 1))
                if pred_all is None:
                    # pred_all = pred_multiply

                    pred_all = pred_multiply_numpy
                else:
                    # pred_all = torch.cat([pred_all, pred_multiply], dim=1)
                    # 以上出来恢复数据为numpy形式时
                    pred_all = np.concatenate((pred_all, pred_multiply_numpy), axis=1)

    return pred_all  # 维度为(-1, od的个数) 为tensor类型 出来要踢掉其梯度！！！！！！！！！！


# 加上一个求解error的函数
def get_error_item(data, od_right_attribute_position, left_position):
    """
    求解误差项的函数
    要进入判别器
    :param left_position: 左边属性的位置，要进入预训练的！！
    :param data:该数据为real data， 或者 fake data
    :param od_right_attribute_position:  右边属性的位置
    :return:
    """
    # 现在先获得左边属性进入预训练的数据

    """gan出来的数据集不需要动"""
    # # 开始进行结合，要遍历该字典
    # 获取gan_out_data的数据，并去掉其梯度。。。。
    data_numpy = data.cpu().detach().numpy()

    pred_all = None
    for od_id, od_id_left_position in left_position.items():
        if len(od_id_left_position) == 1:
            if od_id == '0':
                """
                表明为FD
                """
                fd_left_data_single = None
                for i in od_id_left_position:
                    fd_left_data_single = data_numpy[:, i]
                    fd_left_data_single = np.reshape(fd_left_data_single, (-1, 1))

                # 再封装回tensor里面
                fd_left_data_single = torch.from_numpy(fd_left_data_single)
                fd_left_data_single = torch.tensor(fd_left_data_single, dtype=torch.float32).cuda()

                load_fd_model = "../fd_model_save/fd_model.pkl"
                fd_net = torch.load(load_fd_model)
                fd_net.cuda()
                pred_fd = fd_net(fd_left_data_single)
                """
                 由于不符合transfomer使用的GPU全部是要CPU
                """
                pred_fd_numpy = pred_fd.cpu().detach().numpy()
                # 修改维度
                pred_fd_numpy = np.reshape(pred_fd_numpy, (-1, 1))

                if pred_all is None:
                    # pred_all = pred_fd
                    # 以上为numpy形式时
                    pred_all = pred_fd_numpy
                else:
                    # pred_all = torch.cat([pred_all, pred_fd], dim=1)  # 后面出来要去掉其梯度
                    pred_all = np.concatenate((pred_all, pred_fd_numpy), axis=1)

            else:
                """
                表示这个od为单属性
                """
                od_left_data_single = None
                for i in od_id_left_position:
                    od_left_data_single = data_numpy[:, i]
                    od_left_data_single = np.reshape(od_left_data_single, (-1, 1))

                # 再封装回tensor里面
                od_left_data_single = torch.from_numpy(od_left_data_single)
                od_left_data_single = torch.tensor(od_left_data_single, dtype=torch.float32).cuda()

                # 获取完左边属性的数据之后进行获取网络结构进行融合
                od_left_data_single = od_left_data_single.view(-1, 20, 1)

                load_single_name = "../order_model_single/order_model_single" + str(od_id) + ".pkl"
                net1 = torch.load(load_single_name)
                net1.cuda()
                pred_single = net1(od_left_data_single)  # （-1， 1） 为tensor的形式

                """
                 由于不符合transfomer使用的GPU全部是要CPU
                """
                pred_single_numpy = pred_single.cpu().detach().numpy()
                # 修改维度
                pred_single_numpy = np.reshape(pred_single_numpy, (-1, 1))
                if pred_all is None:
                    # pred_all = pred_single
                    # 以上为numpy形式时
                    pred_all = pred_single_numpy
                else:
                    # pred_all = torch.cat([pred_all, pred_single], dim=1)  # 后面出来要去掉其梯度
                    # 以上出来恢复数据为numpy形式时
                    pred_all = np.concatenate((pred_all, pred_single_numpy), axis=1)
        else:
            """
            判断为多属性的FD时如何处理
            """
            if od_id == '5':
                fd_left_data_multiply = None
                for i in od_id_left_position:
                    tmp = data_numpy[:, i]
                    tmp = np.reshape(tmp, (-1, 1))
                    if fd_left_data_multiply is None:
                        fd_left_data_multiply = tmp
                    else:
                        fd_left_data_multiply = np.concatenate((fd_left_data_multiply, tmp), axis=1)

                # 然后再变为tensor送回模型中去
                fd_left_data_multiply = torch.from_numpy(fd_left_data_multiply).float()
                # 变成三维进入transfomer
                # fd_left_data_multiply = fd_left_data_multiply.unsqueeze(-1)

                # 加载多属性的网络
                load_fd_mul_model = "../fd_model_save/transformer_fd_model.pkl"

                mul_fd_net = torch.load(load_fd_mul_model)
                mul_fd_net.cuda()
                pred_fd_multiply = mul_fd_net(fd_left_data_multiply)

                """
                  由于不符合transfomer使用的GPU全部是要CPU
                """
                pred_fd_multiply_numpy = pred_fd_multiply.detach().numpy()
                # 修改维度
                pred_fd_multiply_numpy = np.reshape(pred_fd_multiply_numpy, (-1, 1))
                if pred_all is None:
                    # pred_all = pred_fd_multiply

                    pred_all = pred_fd_multiply_numpy
                else:
                    # 以上出来恢复数据为numpy形式时
                    pred_all = np.concatenate((pred_all, pred_fd_multiply_numpy), axis=1)

            else:
                """
                   则为多属性的od。处理如下
                """
                od_left_data_multiply = None
                for i in od_id_left_position:
                    tmp = data_numpy[:, i]
                    tmp = np.reshape(tmp, (-1, 1))
                    if od_left_data_multiply is None:
                        od_left_data_multiply = tmp
                    else:
                        od_left_data_multiply = np.concatenate((od_left_data_multiply, tmp), axis=1)

                # 然后再变为tensor送回模型中去
                od_left_data_multiply = torch.from_numpy(od_left_data_multiply)
                od_left_data_multiply = torch.tensor(od_left_data_multiply, dtype=torch.float32).cuda()

                # 再进行转化成三维的
                od_left_data_multiply = od_left_data_multiply.view(-1, 20, len(od_id_left_position))  # 左边属性的维度

                # 获取多属性保存好的网络
                load_multiply_name = "../order_model_multiply/order_model_multiply" + str(od_id) + ".pkl"
                net2 = torch.load(load_multiply_name)
                net2.cuda()
                pred_multiply = net2(od_left_data_multiply)

                """
                由于不符合transfomer使用的GPU全部是要CPU
                """
                pred_multiply_numpy = pred_multiply.cpu().detach().numpy()
                # 修改维度
                pred_multiply_numpy = np.reshape(pred_multiply_numpy, (-1, 1))
                if pred_all is None:
                    # pred_all = pred_multiply

                    pred_all = pred_multiply_numpy
                else:
                    # pred_all = torch.cat([pred_all, pred_multiply], dim=1)
                    # 以上出来恢复数据为numpy形式时
                    pred_all = np.concatenate((pred_all, pred_multiply_numpy), axis=1)
    # 这样就得到通过预训练出来的右属性的值
    # 与原始的数据进行进行相减，所以要

    out_data = None
    for i in od_right_attribute_position:
        tmp = data_numpy[:, i, np.newaxis]

        if out_data is None:
            out_data = tmp
        else:
            out_data = np.concatenate((out_data, tmp), axis=1)

    # 然后相减可以得到label
    error_item = np.fabs(out_data - pred_all)

    return error_item
