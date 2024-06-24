import math
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
# from models import resnet4GCN
from collections import OrderedDict
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, image_size, patch_size, stride=1, padding=1, kernel_size=3, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.image_size = image_size
        self.patch_size = patch_size
        # self.weight = Parameter(torch.FloatTensor(in_features, out_features))

        #选择在图卷积中全连接层替换为卷积操作
        # self.proj = nn.Conv2d(
        #     in_features, out_features,
        #     kernel_size=kernel_size,
        #     stride=stride,
        #     padding=padding)

        self.proj = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(
                in_features,
                out_features,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                bias=False,
                groups=in_features  #  or   2 int(in_features/8)
            )),
            # ('bn', nn.BatchNorm2d(in_features)),
            # ('conv2', nn.Conv2d(
            #     out_features, out_features,
            #     kernel_size=1,
            # )),
            ('bn', nn.BatchNorm2d(in_features)),
            ('relu', nn.GELU()),

        ]))

        # self.dim = in_features
        # self.pooling = nn.AvgPool2d(
        #         kernel_size=kernel_size,
        #         padding=1,
        #         stride=stride,
        #         ceil_mode=True
        #         )

        # if bias:
        #     self.bias = Parameter(torch.FloatTensor(out_features))
        # else:
        #     self.register_parameter('bias', None)
        # self.reset_parameters()

    # def reset_parameters(self):
        # stdv = 1. / math.sqrt(self.weight.size(1))
        # self.weight.data.uniform_(-stdv, stdv)
        # if self.bias is not None:
        #     self.bias.data.uniform_(-stdv, stdv)

    def norm(self, adj):
        # adj = adj + torch.eye(adj.size(1)).expand(adj.size(0), -1, -1).cuda()   # 为每个结点增加自环

        degree = torch.pow(torch.einsum('ijk->ij', [adj]), -0.5)

        degree_b = torch.eye(degree.size(1)).to(device)
        degree_c = degree.unsqueeze(2).expand(*degree.size(), degree.size(1)).to(device)
        degree_diag = degree_c * degree_b
        norm_adj = degree_diag.mul(adj).mul(degree_diag).to(device)

        return norm_adj  # norm_adj

    def forward(self, input, adj):

        # 选择在图卷积中全连接层替换为卷积操作
        # (1)
        # support = torch.matmul(input, self.weight)

        # (2)
        self.h = self.image_size // self.patch_size
        self.w = self.h

        norm_adj = self.norm(adj)
        input = torch.matmul(norm_adj, input)

        input = rearrange(input, 'b (h w) c -> b c h w', h=self.h, w=self.w)

        support = self.proj(input)

        # support = self.pooling(support)  #？？？？？？？

        # support = F.avg_pool2d(support, 2)

        output = rearrange(support, 'b c h w -> b (h w) c', h=self.h, w=self.w)
        #==========================================================================#

        # if self.bias is not None:
        #     return output + self.bias
        # else:
        #
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
