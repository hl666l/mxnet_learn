import d2lzh as d2l
from mxnet import nd
from mxnet.gluon import loss as gloss

"""
定义模型参数
"""
batch_size = 256
"""
fashion_mnist, 是28x28的图片，2828 = 784.故输入是784
总共是10个类别。

"""
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)  # 加载数据
num_inputs, num_outputs, num_hiddens = 784, 10, 256  # 输入数， 输出数， 隐藏层

w1 = nd.random.normal(scale=0.01, shape=(num_inputs, num_hiddens))  # 随机生成权重w1，第一层计算的权重
b1 = nd.zeros(num_hiddens)  # 生成偏置
w2 = nd.random.normal(scale=0.01, shape=(num_hiddens, num_outputs))   # 随机生成权重w2，第二层计算时的权重
b2 = nd.zeros(num_outputs)  # 生成偏置
params = [w1, b1, w2, b2]

for param in params:
    param.attach_grad()


def relu(x):
    """
    定义激活函数
    :param x:
    :return:
    """
    return nd.maximum(x, 0)


def net(x):
    """
    定义模型
    :param x:
    :return:
    """
    x = x.reshape((-1, num_inputs))  # 将图片拉直
    h = relu(nd.dot(x, w1) + b1)
    return nd.dot(h, w2) + b2


# 定义损失函数
loss = gloss.SoftmaxCrossEntropyLoss()

# 训练模型
num_epochs, lr = 5, 0.5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)
