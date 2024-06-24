import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, image_size, patch_size, stride, padding, kernel_size, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid, image_size, patch_size, stride, padding, kernel_size)
        # self.gc2 = GraphConvolution(nfeat, nhid, image_size, patch_size, stride, padding, kernel_size)
        # self.gc3 = GraphConvolution(nfeat, nhid, image_size, patch_size, stride, padding, kernel_size)
        self.dropout = dropout
        # self.linear = nn.Linear(nfeat, nhid)

    def forward(self, x, adj):
        x = F.gelu(self.gc1(x, adj))    # GELU
        x = F.dropout(x, self.dropout, training=self.training)
        # x = F.relu(self.linear(x))
        # x = self.gc2(x, adj)
        #
        # x = F.relu(x)
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = self.gc3(x, adj)

        return x
        # return F.log_softmax(x, dim=1)
