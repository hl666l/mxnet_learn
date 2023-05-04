import d2lzh as d2l
from mxnet import autograd, gluon, nd
from mxnet.gluon import data as gdata, loss as gloss, nn

"""
使用如下函数生成数据集
y = 1.2x - 3.4x**2 + 5.6x**3 + 5 + c
"""
# 初始化 训练数据个数， 测试数据个数， 真实权重值，真实偏置值
n_train, n_test, true_w, true_b = 100, 100, [1.2, -3.4, 5.6], 5
features = nd.random.normal(shape=(n_train + n_test, 1))

#  nd.power(features,2)是对features做平方计算
poly_features = nd.concat(features, nd.power(features, 2), nd.power(features, 3))
labels = (true_w[0] * poly_features[:, 0] + true_w[1] * poly_features[:, 1]
          + true_w[2] * poly_features[:, 2] + true_b)
labels += nd.random.normal(scale=0.1, shape=labels.shape)  # 标签加上随机生成的噪声

# 制作数据集，前100条数据用于训练，后一百条数据用于测试
train_features, test_features, train_labels, test_labels = poly_features[:100], poly_features[100:], labels[:100], labels[100:]
# 定义模型，训练和测试模型， 损失函数
num_epochs, loss = 100, gloss.L2Loss()


def fit_and_plot(train_features, test_features, train_labels, test_labels):
    """
    :param train_features:
    :param test_features:
    :param train_labels:
    :param test_labels:
    :return:
    """
    net = nn.Sequential()  # 跟pytorch中的nn.Sequential作用一样
    net.add(nn.Dense(1))  # 添加一个操作
    net.initialize()  # 必须要初始化
    batch_size = min(10, train_labels.shape[0])  # 判定输入的batch是否比10个少，若少于10个则按实际batch大小。
    # 制作训练的数据集每个batch大小是10
    train_iter = gdata.DataLoader(gdata.ArrayDataset(train_features, train_labels), batch_size, shuffle=True)
    # 注册优化器
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01})
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for x, y in train_iter:
            with autograd.record():
                l = loss(net(x), y)
            l.backward()
            trainer.step(batch_size)  # 更新
        train_ls.append(loss(net(train_features), train_labels).mean().asscalar())  # 计算并记录每轮训练后的平均损失
        test_ls.append(loss(net(test_features), test_labels).mean().asscalar())  # 计算并记录每轮训练后的测试平均损失
    print('final epoch : train loss', train_ls[-1], 'test loss', test_ls[-1])
    print('weight:', net[0].weight.data().asnumpy(),
          '\nbias:', net[0].bias.data().asnumpy())


fit_and_plot(train_features, test_features, train_labels, test_labels)
