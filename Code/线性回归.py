import matplotlib
import matplotlib_inline
from IPython import display
from matplotlib import pyplot as plt
from mxnet import autograd, nd
import random

"""
生成数据集
"""
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)
print(features[0], labels[0])


def use_svg_display():
    matplotlib_inline.backend_inline.set_matplotlib_formats()
    # display.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


set_figsize()
plt.scatter(features[:, 1].asnumpy(), labels.asnumpy(), 1);
