import theano
import theano.tensor as T
from theano.tensor.signal import pool



import numpy as np
import pylab
from PIL import Image


# instantiate 4D tensor for input
input = T.tensor4(name='input')

# initialize shared variable for weights.
w = np.array([[0, 1, 1],[1, 0, 1], [1, 1, 0]])
W = theano.shared(w.reshape(1, 1, 3, 3), theano.config.floatX)

# initialize shared variable for bias (1D tensor) with random values
# IMPORTANT: biases are usually initialized to zero. However in this
# particular application, we simply apply the convolutional layer to
# an image without learning the parameters. We therefore initialize
# them to random values to "simulate" learning.
b = theano.shared(np.array([0.05]), theano.config.floatX)

# build symbolic expression that computes the convolution of input with filters in w

# build symbolic expression to add bias and apply activation function, i.e. produce neural net layer output
# A few words on ``dimshuffle`` :
#   ``dimshuffle`` is a powerful tool in reshaping a tensor;
#   what it allows you to do is to shuffle dimension around
#   but also to insert new ones along which the tensor will be
#   broadcastable;
#   dimshuffle('x', 2, 'x', 0, 1)
#   This will work on 3d tensors with no broadcastable
#   dimensions. The first dimension will be broadcastable,
#   then we will have the third dimension of the input tensor as
#   the second of the resulting tensor, etc. If the tensor has
#   shape (20, 30, 40), the resulting tensor will have dimensions
#   (1, 40, 1, 20, 30). (AxBxC tensor is mapped to 1xCx1xAxB tensor)
#   More examples:
#    dimshuffle('x') -> make a 0d (scalar) into a 1d vector
#    dimshuffle(0, 1) -> identity
#    dimshuffle(1, 0) -> inverts the first and second dimensions
#    dimshuffle('x', 0) -> make a row out of a 1d vector (N to 1xN)
#    dimshuffle(0, 'x') -> make a column out of a 1d vector (N to Nx1)
#    dimshuffle(2, 0, 1) -> AxBxC to CxAxB
#    dimshuffle(0, 'x', 1) -> AxB to Ax1xB
#    dimshuffle(1, 'x', 0) -> AxB to Bx1xA

u = T.nnet.conv2d(input, W) + b.dimshuffle('x', 0, 'x', 'x')
y = T.nnet.sigmoid(u)
o = pool.pool_2d(y, (2,2), mode='max')

# create theano function to compute filtered images
f = theano.function([input], [u, y, o])

# open random image of dimensions 639x516
I = np.array([[0.5, -0.1, 0.2, 0.3, 0.5],
              [0.8, 0.1, -0.5, 0.5, 0.1],
              [-1.0, 0.2, 0.0, 0.3, -0.2],
              [0.7, 0.1, 0.2, -0.6, 0.3],
              [-0.4, 0.0, 0.2, 0.3, -0.3]])


# put image in 4D tensor of shape (1, 3, height, width)
uu, yy, oo = f(I.reshape(1, 1, 5, 5))

print(uu)
print(yy)
print(oo)
