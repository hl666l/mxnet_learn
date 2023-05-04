"""
不含模型参数的自定义层
"""
from mxnet import gluon, nd
from mxnet.gluon import nn


class CenteredLayer(nn.Block):
    def __init__(self):
        super(CenteredLayer, self).__init__()

    def forward(self, x):
        return x - x.mean()


layer = CenteredLayer()
print(layer(nd.array([1, 2, 3, 4, 5])))
