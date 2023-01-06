#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Xiang

from pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation
import torch
from torch import nn
import math
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.model = nn.Sequential()
        for i, num_channels in enumerate(dims[:-1]):
            self.model.add_module('fc_%d' % (
                i+1), nn.Linear(num_channels, dims[i+1]))
            if i != len(dims) - 2:
                self.model.add_module('relu_%d' % (i+1), nn.ReLU())

    def forward(self, features):
        return self.model(features)


class MLPConv(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.model = nn.Sequential()
        for i, num_channels in enumerate(dims[:-1]):
            self.model.add_module('conv1d_%d' % (
                i+1), nn.Conv1d(num_channels, dims[i+1], kernel_size=1))
            if i != len(dims) - 2:
                self.model.add_module('relu_%d' % (i+1), nn.ReLU())

    def forward(self, inputs):
        return self.model(inputs)


class ContractExpandOperation(nn.Module):
    def __init__(self, num_input_channels, up_ratio):
        super().__init__()
        self.up_ratio = up_ratio
        # PyTorch default padding is 'VALID'
        # !!! rmb to add in L2 loss for conv2d weights
        self.conv2d_1 = nn.Conv2d(num_input_channels, 64, kernel_size=(
            1, self.up_ratio), stride=(1, 1))
        self.conv2d_2 = nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1))
        self.conv2d_3 = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, inputs):  # (32, 64, 2048)
        # (32, 64, 2, 1024)
        net = inputs.view(inputs.shape[0], inputs.shape[1], self.up_ratio, -1)
        net = net.permute(0, 1, 3, 2).contiguous()  # (32, 64, 1024, 2)
        net = F.relu(self.conv2d_1(net))  # (32, 64, 1024, 1)
        net = F.relu(self.conv2d_2(net))  # (32, 128, 1024, 1)
        net = net.permute(0, 2, 3, 1).contiguous()  # (32, 1024, 1, 128)
        # (32, 1024, 2, 64)
        net = net.view(net.shape[0], -1, self.up_ratio, 64)
        net = net.permute(0, 3, 1, 2).contiguous()  # (32, 64, 1024, 2)
        net = F.relu(self.conv2d_3(net))  # (32, 64, 1024, 2)
        net = net.view(net.shape[0], 64, -1)  # (32, 64, 2048)
        return net


class MLP_CONV(nn.Module):
    def __init__(self, in_channel, layer_dims, bn=None):
        super(MLP_CONV, self).__init__()
        layers = []
        last_channel = in_channel
        for out_channel in layer_dims[:-1]:
            layers.append(nn.Conv1d(last_channel, out_channel, 1))
            if bn:
                layers.append(nn.BatchNorm1d(out_channel))
            layers.append(nn.ReLU())
            last_channel = out_channel
        layers.append(nn.Conv1d(last_channel, layer_dims[-1], 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.mlp(inputs)


class Conv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=(1, 1), stride=(1, 1), if_bn=True, activation_fn=torch.relu):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel,
                              kernel_size, stride=stride)
        self.if_bn = if_bn
        self.bn = nn.BatchNorm2d(out_channel)
        self.activation_fn = activation_fn

    def forward(self, input):
        out = self.conv(input)
        if self.if_bn:
            out = self.bn(out)

        if self.activation_fn is not None:
            out = self.activation_fn(out)

        return out


def symmetric_sample(points, num):
    """
    Args
        pcd: (b, 16384, 3)

    returns
        new_pcd: (b, num * 2, 3)
    """
    input_fps = fps_subsample(points, num)

    input_fps_flip = torch.cat(
        [torch.unsqueeze(input_fps[:, :, 0], dim=2), torch.unsqueeze(input_fps[:, :, 1], dim=2),
         torch.unsqueeze(-input_fps[:, :, 2], dim=2)], dim=2)
    input_fps = torch.cat([input_fps, input_fps_flip], dim=1)
    return input_fps


def gen_grid_up(up_ratio, grid_size=0.2):
    sqrted = int(math.sqrt(up_ratio)) + 1
    for i in range(1, sqrted + 1).__reversed__():
        if (up_ratio % i) == 0:
            num_x = i
            num_y = up_ratio // i
            break

    grid_x = torch.linspace(-grid_size, grid_size, steps=num_x)
    grid_y = torch.linspace(-grid_size, grid_size, steps=num_y)

    x, y = torch.meshgrid(grid_x, grid_y)  # x, y shape: (2, 1)
    grid = torch.stack([x, y], dim=-1).view(-1, 2).transpose(0, 1).contiguous()
    return grid


def fps_subsample(pcd, n_points=2048):
    """
    Args
        pcd: (b, 16384, 3)

    returns
        new_pcd: (b, n_points, 3)
    """
    new_pcd = gather_operation(pcd.permute(
        0, 2, 1).contiguous(), furthest_point_sample(pcd, n_points))
    new_pcd = new_pcd.permute(0, 2, 1).contiguous()
    return new_pcd
