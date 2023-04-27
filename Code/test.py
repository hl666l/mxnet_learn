from mxnet import nd, gpu
a = nd.zeros((2, 3), gpu(0))
print(a)
