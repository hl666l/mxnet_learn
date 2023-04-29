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
            test_ls.append(log_rms(net, test_features, test_labels))  # 每训练一个epoch,计算测试误差，并将其添加到list末尾
    return train_ls, test_ls  # 返回训练误差list， 测试误差list


# k折交叉验证
def get_k_fold_data(k, i, x, y):
    """
    :param k:数据分成 K份
    :param i: 第几次交叉，用来决定第几份数据做验证集
    :param x:特征数据
    :param y:标签数据
    :return:  训练集， 测试集
    """
    assert k > 1  # 判断k是否大于1,否则无意义
    fold_size = x.shape[0] // k  # 获取数据分成k份，每份有多少个标本。 // 相除取整操作
    x_train, y_train = None, None  # 用来存储
    for j in range(k):  # 循环切片。当j与 i
        idx = slice(j * fold_size, (j + 1) * fold_size)  # 作切片slice[开始,结束] 返回
        x_part, y_part = x[idx, :], y[idx]  # 保留切割下来的作为测试部分的数据
        if j == i:  # 若j=i则此时的 x_part, y_part是测试部分数据。用x_valid, y_valid
            x_valid, y_valid = x_part, y_part
        elif x_train is None:
            x_train, y_train = x_part, y_part
        else:
            """
            将每次切割的测试集部分连接到x_train后面。当i=j时不连接，用x_valid, y_valid保存。
            这样经过k次切割后x_train, y_train保存的就是除切割部分的其它数据。
            """
            x_train = nd.concat(x_train, x_part, dim=0)
            y_train = nd.concat(y_train, y_part, dim=0)
    return x_train, y_train, x_valid, y_valid


def k_fold(k, x_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, x_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)  # 训练误差 测试误差
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse', range(1, num_epochs + 1), valid_ls,
                         ['train', 'valid'])
            print('fold %d, train rmse %f, valid rmse %f' % (i, train_ls[-1], valid_ls[-1]))
    return train_l_sum / k, valid_l_sum / k


k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64


# 训练并预测
def train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size):
    """
    :param train_features:
    :param test_features:
    :param train_labels:
    :param test_data: test_labels
    :param num_epochs:
    :param lr:
    :param weight_decay: 权重衰减
    :param batch_size:
    :return:
    """
    net = get_net()
    # 训练模型
    train_ls, _ = train(net, train_features, train_labels, None, None, num_epochs, lr, weight_decay, batch_size)
    d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse')
    print('train rmse %f' % train_ls[-1])
    # 用测试数据预测
    preds = net(test_features).asnumpy()
    # 保存测试数据
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    # 将测试数据的ID号与预测结果的连接起来。
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    # 保存成csv文件，不加行号。提交给kaggle网站
    submission.to_csv('submission.csv', index=False)


train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size)
