# @.@ coding  : utf-8 ^_^
# @Author     : Leon Rein
# @Time       : 22/11/21 ~ 11:20:30
# @File       : get_mean_std.py
# @Note       :

import torch


def get_mean_std_value(data_loader):
    """
    It calculates the mean and standard deviation of the data in the given loader over **channels**.
    This function is designed for: 
        4D tensor of [batch_size, channels, height, width] or 3D of [batch_size, channels, length].
    Note that the num of samples should be divisible by batch_size (or mark drop_last=True)!

    :param loader: a DataLoader object that iterates over the dataset
    :return: mean and std
    """

    mean_sum, var_sum, num_batches = 0, 0, 0

    for data, _ in data_loader:
        dims = [0] + list(range(2, data.ndim))  # 总维度除了第一个维度 (channel)
        # 计算 batch 内除了维度 1 以外的均值和，dim=1 为通道数量，不用参与计算
        mean_sum += data.mean(dim=dims)
        # 计算 batch 内除了维度 1 以外的方差，dim=1 为通道数量，不用参与计算
        var_sum += data.var(dim=dims)
        # 统计batch的数量
        num_batches += 1

    mean = mean_sum / num_batches
    std = torch.sqrt(var_sum/num_batches)  # 计算标准差

    return mean, std
