"""
Block类是nn模块提供的一个模型构造类，继承Block来自定义我们自己的模型。
下面继承Block类，构造一个MLP模型，MLP重载了Block中的__init__函数和forward函数。
他们分别用于创建模型参数和定义前向计算。
"""
from mxnet import nd
from mxnet.gluon import nn


class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')  # 定义一个隐藏层，输出是256维，并且使用relu激活函数
        self.output = nn.Dense(10)  # 定义一个全连接层输出是10维

    def forward(self, x):
        return self.output(self.hidden(x))


# test

x = nd.random.uniform(shape=(2, 20))
net = MLP()
net.initialize()
print(net(x))

"""
Sequential类继承Block类

Block类是一个通用的部件， Sequential类继承Block类。
当模型的前向计算为简单的串联各层的计算时，可以通过更简单的方式定义模型。
这正是Sequential类的目的：他提供add函数来逐一添加串联的Block子类实例。

"""

#  使用Sequential思想，通过add函数构建自定义模型
class MySequential(nn.Block):
    def __init__(self):
        super(MySequential, self).__init__()

    def add(self, block):
        self._children[block.name] = block

    def forward(self, x):
        for block in self._children.values():
            x = block(x)
        return x


net = MySequential()
#  添加Block实例来创建net网络模型
net.add(nn.Dense(256, activation='relu'))  # 添加Block实例
net.add(nn.Dense(10))  # 添加Block实例
net.initialize()
print(net(x))

"""
构造更加复杂的模型
"""
class FancyMLP(nn.Block):
    def __init__(self):
        super(FancyMLP, self).__init__()
        # 使用get_constant函数创建的随机权重参数不会在训练中被迭代（即常参数）
        self.rand_weight = self.params.get_constant('rand_weight', nd.random.uniform(shape=(20, 20)))
        self.dense = nn.Dense(20, activation='relu')

    def forward(self, x):
        x = self.dense(x)
        # 使用创建的常参数，以及NDArray的relu函数和dot函数
        x = nd.relu(nd.dot(x, self.rand_weight.data()) + 1)
        # 复用全连接层，等于两个全连接层共享参数
        x = self.dense(x)
        # 控制流，这里使用asscalar()返回标量进行比较
        while x.norm().asscalar() > 1:
            x /= 2
        if x.norm().asscalar()<0.8:
            x *=10
        return x.sum()

net = FancyMLP()
net.initialize()
print(net(x))




