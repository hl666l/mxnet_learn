"""
ReLu(x) = max(x,0)
"""

import matplotlib_inline
import d2lzh as d2l
from mxnet import autograd, nd
from Myfuncation import xyplot

x = nd.arange(-8.0, 8.0, 0.1)
x.attach_grad()
with autograd.record():
    y = x.relu()
xyplot(x, y, 'relu')
