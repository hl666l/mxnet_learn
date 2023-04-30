import mxnet
from mxnet import init, nd
from mxnet.gluon import nn

"""
定义一个简单的网络
mxnet.init中含有很多模型初始化方法
"""
net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()
# net.initialize(ctx=mxnet.gpu(0))
# x = nd.random.uniform(shape=(2, 20), ctx=mxnet.gpu(0))
x = nd.random.uniform(shape=(2, 20))
y = net(x)
# print(y)

"""

访问模型参数

"""

print('第一个输出：',net[0].params, '第一个的第二个输出：',type(net[0].params))
print('第二个输出：',net[0].params['dense0_weight'],'第二个的第二个输出', net[0].weight)

print('第三个输出：',net[0].weight.data())


