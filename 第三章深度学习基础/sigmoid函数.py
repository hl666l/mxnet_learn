from Myfuncation import xyplot
from mxnet import autograd, nd

x = nd.arange(-8.0, 8.0, 0.1)
x.attach_grad()

with autograd.record():
    y = x.sigmoid()
xyplot(x, y, 'sigmoid')
