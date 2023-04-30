from mxnet import init, nd
from mxnet.gluon import nn
class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print("Init", name, data.shape)

net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'),
        nn.Dense(10)
        )
net.initialize(init=MyInit())  # 调用MyInit并未有输出。说明并未真正初始化参数
x = nd.random.uniform(shape=(2, 20))
# 执行一次前向操作，此时才有输出。故真正的初始化是在第一次前向操作时进行的。
# 当我们执行第二前向操作时不会再有输出。故初始化是在第一次前向操作时执行的。
y = net(x)

"""
避免延后初始化。
当系统调用initialize（）时知道输入参数的形状，那么延后初始化就不会出现。
"""

# 第一种情况，对已初始化的模型重新初始化。因为系统已经知道输入参数的形状，则能够立即进行初始化。
net.initialize(init=MyInit(), force_reinit=True)

# 第二种情况，创建网络层时指定输入形状。
# 这样系统就不需要额外的信息推测参数形状。
net = nn.Sequential()
net.add(nn.Dense(256, in_units=20,activation='relu'))
net.add(nn.Dense(10, in_units=256))
net.initialize(init=MyInit())

