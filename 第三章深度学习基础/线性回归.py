import matplotlib
import matplotlib_inline
from IPython import display
from matplotlib import pyplot as plt
from mxnet import autograd, nd
import random

"""
生成数据集
"""
num_inputs = 2  # 每个样本有两个数据
num_examples = 1000  # 样本数量

true_w = [2, -3.4]  # 权重
true_b = 4.2  # 偏差

# y = w1*x + w2*x2 + b + c
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))  # 随机生成全部样本
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b  # 生成标签y
labels += nd.random.normal(scale=0.01, shape=labels.shape)  # 增加一些噪声， 噪声服从均值为0, 标准差为0.01的正态分布。
print(features[0], labels[0])


def use_svg_display():
    matplotlib_inline.backend_inline.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize  # 设置图片尺寸


# set_figsize()
# plt.scatter(features[:, 1].asnumpy(), labels.asnumpy(), 1);


"""
加载数据函数，每次返回随机的十个数据
"""


def data_iter(batch_size, features, labels):
    """
    :param batch_size:
    :param features: 特征数据
    :param labels: 标签
    :return:
    """
    num_examples = len(features)  # 获取特征数据长度
    indices = list(range(num_examples))  # 生成下标列表
    random.shuffle(indices)  # 打乱列表
    for i in range(0, num_examples, batch_size):
        j = nd.array(indices[i: min(i + batch_size, num_examples)])  # 取数据
        yield features.take(j), labels.take(j)  # 返回数据，和标签


batch_size = 10
# 测试一下数据加载函数
for x, y in data_iter(batch_size, features, labels):
    print(x, y)
    break
#  初始化我们的权重，偏置，噪声
w = nd.random.normal(scale=0.01, shape=(num_inputs, 1))
b = nd.zeros(shape=(1,))
# 声明我们需要计算那几个变量的梯度
w.attach_grad()
b.attach_grad()

"""
定义模型
"""


def linreg(x, w, b):
    return nd.dot(x, w) + b


"""
定义损失函数
"""


def squared_loss(y_hat, y):
    """
    :param y_hat: 预测值
    :param y: 真实值，标签
    :return:
    """
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


"""
定义优化算法
"""


def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size


"""
训练模型
"""

lr = 0.03
num_epochs = 10
net = linreg
loss = squared_loss
for epoch in range(num_epochs):
    for x, y in data_iter(batch_size, features, labels):
        with autograd.record():
            l = loss(net(x, w, b), y)
        l.backward()  # 计算函数l的梯度
        sgd([w, b], lr, batch_size)
    train_l = loss(net(features, w, b), labels)
    print('epoch%d, loss%f' % (epoch + 1, train_l.mean().asnumpy()))
