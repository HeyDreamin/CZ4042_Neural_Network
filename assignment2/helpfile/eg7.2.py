import theano
from theano import tensor as T
import numpy as np
from load import mnist
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool
import matplotlib.pyplot as plt



trX, teX, trY, teY = mnist(onehot=True)

trX = trX.reshape(-1, 1, 28, 28)
teX = teX.reshape(-1, 1, 28, 28)


X = T.ftensor4('X')
Y = T.fmatrix('Y')

w = np.array([[[-1, 0, 1],[-2, 0, 2], [-1, 0, 1]],
     [[1, 2, 1],[0, 0, 0], [-1, -2, -1]],
     [[3, 4, 3], [4, 5, 4], [3, 4, 3]]])
W = theano.shared(np.asarray(w.reshape(3, 1, 3, 3), dtype = X.type))

u = conv2d(X, W)
y = T.nnet.sigmoid(u)
o = pool.pool_2d(y, (2, 2), mode='average_exc_pad')

maps = theano.function([X], [u, y, o], allow_input_downcast = True)

ind = np.random.randint(low=0, high=60000)
img = trX[ind:ind+1]
uu, yy, oo = maps(img)

print(img.shape)

plt.figure()
plt.gray()
plt.axis('off')
plt.imshow(img[0,0,:,:])
plt.savefig('figure_7.2_1.png')

plt.figure()
plt.gray()
plt.subplot(3,1,1), plt.axis('off'), plt.imshow(uu[0,0,:,:])
plt.subplot(3,1,2), plt.axis('off'), plt.imshow(uu[0,0,:,:])
plt.subplot(3,1,3), plt.axis('off'), plt.imshow(uu[0,0,:,:])
plt.savefig('figure_7.2_2.png')

plt.figure()
plt.gray()
plt.subplot(3,1,1), plt.axis('off'), plt.imshow(yy[0,0,:,:])
plt.subplot(3,1,2), plt.axis('off'), plt.imshow(yy[0,0,:,:])
plt.subplot(3,1,3), plt.axis('off'), plt.imshow(yy[0,0,:,:])
plt.savefig('figure_7.2_3.png')

plt.figure()
plt.gray()
plt.subplot(3,1,1), plt.axis('off'), plt.imshow(oo[0,0,:,:])
plt.subplot(3,1,2), plt.axis('off'), plt.imshow(oo[0,0,:,:])
plt.subplot(3,1,3), plt.axis('off'), plt.imshow(oo[0,0,:,:])
plt.savefig('figure_7.2_4.png')

plt.show()

