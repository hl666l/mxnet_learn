"""
mxnet-cu117==1.9.1
numpy==1.20
pandas==1.1.0
"""
import d2lzh as d2l
from mxnet import autograd, init, nd, gluon
from mxnet.gluon import data as gdata, loss as gloss, nn
import numpy as np
import pandas as pd

# 读取数据
train_data = pd.read_csv('/home/helei/PycharmProjects/mxnet_learn/data/train.csv')
test_data = pd.read_csv('/home/helei/PycharmProjects/mxnet_learn/data/test.csv')

# print(len(train_data))
# print(len(test_data))
#  concat()默认是按照行连接。对所有数据进连接起来
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
"""
预处理所有特征数据
"""

# 获取所有数值类型特征值的列标
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
# 对所有的数值特征减去该列的均值，再除以标准差。获得标准化后的特征值。
all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
# 标准化后列的均值是0, 故用0填充缺失的数值
all_features[numeric_features] = all_features[numeric_features].fillna(0)

"""
将离散值转成数值特征。例如一列中分别有三中数据 FA，FB，NAN。
那么我们将FA，FB，NAN转换成三种列特征，有FA的行在FA列上的值为1,否则为0。以此类推到FB，NAN列上。这样会增加原数据列数。
dummy_na=True 表示将缺失值也作为合法输入
"""
all_features = pd.get_dummies(all_features, dummy_na=True)

n_train = train_data.shape[0]
train_features = nd.array(all_features[:n_train].values)
test_features = nd.array(all_features[n_train:].values)
train_labels = nd.array(train_data.SalePrice.values).reshape((-1, 1))

# 训练模型
loss = gloss.L2Loss()


def get_net():
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize()
    return net


def log_rms(net, features, labels):
    clipped_preds = nd.clip(net(features), 1, float('inf'))
    rmse = nd.sqrt(2 * loss(clipped_preds.log(), labels.log()).mean())
    return rmse.asscalar()


def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []  # 训练损失，测试损失。
    # 制作DataLoader类型数据集
    train_iter = gdata.DataLoader(gdata.ArrayDataset(train_features, train_labels), batch_size, shuffle=True)
    # 初始化训练的优化函数Adam优化算法，学习率，权重
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': learning_rate, 'wd': weight_decay})
    for epoch in range(num_epochs):
        for x, y in train_iter:
            with autograd.record():  # 在autograd.record()作用域下定义函数有助于自动求梯度。
                l = loss(net(x), y)
            l.backward()
            trainer.step(batch_size)
        train_ls.append(log_rms(net, train_features, train_labels))  # 一个epoch训练完之后的，将所有训练数据扔进网络求对数均方根误差
        if test_labels is not None:
            test_ls.append(log_rms(net, test_features, test_labels))  # 每训练一个epoch,计算测试误差
    return train_ls, test_ls  # 返回训练误差list， 测试误差list

# k折交叉验证
