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
# net[0].params 获取第一层网络的权重以及偏置的的信息
print('第一个输出：', net[0].params, '第一个的第二个输出：', type(net[0].params))
# 这两种方式获取的都是网络第一层的权重。
# Parameter dense0_weight (shape=(256, 20), dtype=float32)
print('第二个输出：', net[0].params['dense0_weight'], '第二个的第二个输出', net[0].weight)
# 输出第一层网络权重的具体数值，偏置的具体数值。
print('第三个输出：', net[0].weight.data(), net[0].bias.data())

# 会输出所有层的权重偏置信息
"""
通过collect_params()函数
来获取net变量所有嵌套的层所包含的所有参数。
会输出所有层的权重偏置信息
sequential0_ (
  Parameter dense0_weight (shape=(256, 20), dtype=float32)
  Parameter dense0_bias (shape=(256,), dtype=float32)
  Parameter dense1_weight (shape=(10, 256), dtype=float32)
  Parameter dense1_bias (shape=(10,), dtype=float32)
)
"""
print(net.collect_params())
"""
collect_params()函数可以通过正则表达式来匹配参数名，
从而筛选出需要的参数
"""
print(net.collect_params('.*weight'))

""""
初始化模型参数
"""
# 例子 我们将权重参数初始化成均值为0,
# 标准差为0.01的正态分布随机数， 并依然将偏差参数清零。
# 非首次对模型初始化需要指定force_reinit为True
net.initialize(init=init.Normal(sigma=0.01), force_reinit=True)
print(net[0].weight.data()[0])

# 使用常数初始化权重
net.initialize(init=init.Constant(1), force_reinit=True)
print(net[0].weight.data()[0])

"""
自定义初始化方法
"""


#  下面的例子我们令权重有一半概率初始化为0,
#  另一半概率初始化为[-10,-5]和[5,10]两个区间均匀分布的随机数
class MyInit(init.Initializer):
    def _init_weight(self, name, data):  # 继承init.Initializer后重写_init_weight方法初始化权重参数
        print('Init', name, data.shape)
        data[:] = nd.random.uniform(low=0, high=10, shape=data.shape)
        data *= data.abs() >= 5

    def _init_bias(self, _, arr):  # 初始化偏置
        pass


#  用我们自定义的初始化操作，初始化模型。
net.initialize(MyInit(), force_reinit=True)
print(net[0].weight.data()[0])
#  还可以通过Parameter类的set_data函数来直接改写模型参数。
# 下面我们将隐藏层参数在现有的基础上加1
net[0].weight.set_data(net[0].weight.data() + 1)
print(net[0].weight.data()[0])

"""
共享模型参数
"""
#  声明net是顺序执行类网络模型
net = nn.Sequential()
shared = nn.Dense(8, activation='relu')
net.add(nn.Dense(8, activation='relu'),
        shared,
        nn.Dense(8, activation='relu', params=shared.params),  # 将shared层的参数共享给这一层
        nn.Dense(10)
        )
net.initialize()
x = nd.random.uniform(shape=(2, 20))
net(x)
print(net[1].weight.data()[0] == net[2].weight.data()[0])  # 判断两层参数是否相等



















