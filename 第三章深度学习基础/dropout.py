"""
丢弃法，就是把权重随机赋值为零。这样这个权重的梯度就为零。
好处是增强模型的鲁棒性。
"""
import d2lzh as d2l
from mxnet import autograd, nd, init, gluon
from mxnet.gluon import loss as gloss, nn


def dropout(x, drop_prob):
    assert 0 <= drop_prob <= 1
    keep_prob = 1 - drop_prob
    if keep_prob == 0:
        return x.zeros_like()
    mask = nd.random.uniform(0, 1, x.shape) < keep_prob
    return mask * x / keep_prob


"""

几个例子分别是丢弃率：0, 0.5, 0.8

"""

x = nd.arange(16).reshape((2, 8))
"""
[[ 0.  1.  2.  3.  4.  5.  6.  7.]
 [ 8.  9. 10. 11. 12. 13. 14. 15.]]
<NDArray 2x8 @cpu(0)>
"""
print(dropout(x, 0))
"""
[[ 0.  2.  4.  6.  0.  0.  0. 14.]
 [ 0. 18.  0.  0. 24. 26. 28.  0.]]
<NDArray 2x8 @cpu(0)>
"""
print(dropout(x, 0.5))
"""
[[ 0.  0.  0.  0.  0. 25.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.]]
<NDArray 2x8 @cpu(0)>
"""
print(dropout(x, 0.8))
