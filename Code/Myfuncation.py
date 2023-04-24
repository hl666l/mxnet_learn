import matplotlib_inline
import d2lzh as d2l
from mxnet import autograd, nd


def xyplot(x_vals, y_vals, name):
    """
    画图函数
    :param x_vals:x轴
    :param y_vals:y轴
    :param name: 图像title
    :return: NoNe
    """
    d2l.set_figsize(figsize=(5, 2.5))
    d2l.plt.plot(x_vals.asnumpy(), y_vals.asnumpy())
    d2l.plt.xlabel('x')
    d2l.plt.ylabel(name + '(x)')
    d2l.plt.show()
