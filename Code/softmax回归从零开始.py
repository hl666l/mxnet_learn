import matplotlib_inline
import d2lzh as d2l
from mxnet import autograd, nd

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)  # 制作数据集 默认的数据就是fashion-mnist
#  输入样本数， 输出种类
num_inputs = 784
num_outputs = 10
#  随机生成 权重w, 偏置b
w = nd.random.normal(scale=0.01, shape=(num_inputs, num_outputs))
b = nd.zeros(num_outputs)
#  声明要计算的梯度的变量
w.attach_grad()
b.attach_grad()


def softmax(x):
    """
    构造softmax函数
    :param x:
    :return: 返回的是batch_size行元素，每行有num_outputs个元素

    对矩阵做指数运算得x_exp,然后求x_exp每行的和，
    返回每行各个元素除每行的和：x_exp/ partition。
    这保证了矩阵每行元素和为1,且不为负
    """
    x_exp = x.exp()
    partition = x_exp.sum(axis=1, keepdims=True)
    return x_exp / partition


def net(x):
    """
    定义模型
    :param x:
    :return: 返回batch_size行 [1,2,3,4,5,6,7,8,9,10]
    注意x.reshape((-1, num_inputs))，-1是因为此时 x 的值是batch_size个num_inputs,
    但是若最后一个batch数目不够batch_size，则会出错。故用-1,让系统判断该取何值。
    """
    return softmax(nd.dot(x.reshape((-1, num_inputs)), w) + b)


def cross_entropy(y_hat, y):
    """
    定义损失函数
    :param y_hat:
    :param y:
    :return: 交叉熵损失函数
    """
    return -nd.pick(y_hat, y).log()


def accuracy(y_hat, y):
    """
    计算分类准确率
    :param y_hat:
    :param y:
    :return:
    y_hat.argmax(axis=1) 返回这一行最大值的索引。
    可以理解成，预测值为一行，这行有num_outputs 个预测值。
    通过判断最大值的下标确定哪个种类的可能性最大

    因为(y_hat.argmax(axis=1) == y.astype('float32'))返回的是0,1为元素的标量，
    mean()算这个标量的平均值，就是准确率
    """
    return (y_hat.argmax(axis=1) == y.astype('float32')).mean().asscalar()


def evaluate_accuracy(data_iter, net):
    """
    判断在一整个数据集上的准确率
    :param data_iter:
    :param net:
    :return:
    """
    acc_sum, n = 0.0, 0
    for x, y in data_iter:
        y = y.astype('float32')
        acc_sum += (net(x).argmax(axis=1) == y).sum().asscalar()
        n += y.size
    return acc_sum / n


num_epochs, lr = 5, 0.1


def tain_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None, trainer=None):
    """
    :param net:
    :param train_iter:
    :param test_iter:
    :param loss:
    :param num_epochs:
    :param batch_size:
    :param params:初始化的w,b值
    :param lr:
    :param trainer:
    :return:
    """
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for x, y in train_iter:
            with autograd.record():
                y_hat = net(x)
                l = loss(y_hat, y).sum()
            l.backward()
            if trainer is None:
                d2l.sgd(params, lr, batch_size)
            else:
                trainer.step(batch_size)
            y = y.astype('float32')
            train_l_sum += l.asscalar()
            train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()  # asscalar()将向量转成标量
            n += y.size
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train_acc_sum %.3f, test_acc %.3f' % (epoch + 1,
                                                                          train_l_sum / n, train_acc_sum / n, test_acc))


tain_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [w, b], lr)
