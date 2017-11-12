from load import mnist
import numpy as np
from matplotlib import pyplot as plt

import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool
 
# np.random.seed(10)
batch_size = 128
noIters = 100

def init_weights_bias4(filter_shape, d_type):
	fan_in = np.prod(filter_shape[1:])
	fan_out = filter_shape[0] * np.prod(filter_shape[2:])
	 
	bound = np.sqrt(6. / (fan_in + fan_out))
	w_values =  np.asarray(
			np.random.uniform(low=-bound, high=bound, size=filter_shape),
			dtype=d_type)
	b_values = np.zeros((filter_shape[0],), dtype=d_type)
	return theano.shared(w_values,borrow=True), theano.shared(b_values, borrow=True)

def init_weights_bias2(filter_shape, d_type):
	fan_in = filter_shape[1]
	fan_out = filter_shape[0]
	 
	bound = np.sqrt(6. / (fan_in + fan_out))
	w_values =  np.asarray(
			np.random.uniform(low=-bound, high=bound, size=filter_shape),
			dtype=d_type)
	b_values = np.zeros((filter_shape[1],), dtype=d_type)
	return theano.shared(w_values,borrow=True), theano.shared(b_values, borrow=True)

def set_weights_bias4(filter_shape, d_type, w, b):
	fan_in = np.prod(filter_shape[1:])
	fan_out = filter_shape[0] * np.prod(filter_shape[2:])
	 
	bound = np.sqrt(6. / (fan_in + fan_out))
	w_values =  np.asarray(
			np.random.uniform(low=-bound, high=bound, size=filter_shape),
			dtype=d_type)
	b_values = np.zeros((filter_shape[0],), dtype=d_type)
	w.set_value(w_values), b.set_value(b_values)
	return

def set_weights_bias2(filter_shape, d_type, w, b):
	fan_in = filter_shape[1]
	fan_out = filter_shape[0]
	 
	bound = np.sqrt(6. / (fan_in + fan_out))
	w_values =  np.asarray(
			np.random.uniform(low=-bound, high=bound, size=filter_shape),
			dtype=d_type)
	b_values = np.zeros((filter_shape[1],), dtype=d_type)
	w.set_value(w_values), b.set_value(b_values)
	return

def model(X, w1, b1, w2, b2, w3, b3, w4, b4): 
	pool_dim = (2, 2)
	# First convolution layer C1 and pooling layer S1
	y1 = T.nnet.relu(conv2d(X, w1) + b1.dimshuffle('x', 0, 'x', 'x'))
	o1 = pool.pool_2d(y1, pool_dim, ignore_border=True)
	
	# Second convolution layer C2 and pooling layer S2
	y2 = T.nnet.relu(conv2d(o1, w2) + b2.dimshuffle('x', 0, 'x', 'x'))
	o2 = pool.pool_2d(y2, pool_dim, ignore_border=True)

	# Input for fully connected layer
	fcinput = T.flatten(o2, outdim=2)
	# Fully connected layer F3 of size 100
	fcoutput = T.nnet.relu(T.dot(fcinput, w3) + b3)

	# Softmax layer F4 of size 10
	pyx = T.nnet.softmax(T.dot(fcoutput, w4) + b4)
	return y1, o1, y2, o2, pyx

def shuffle_data (samples, labels):
	idx = np.arange(samples.shape[0])
	np.random.shuffle(idx)
	samples, labels = samples[idx], labels[idx]
	return samples, labels

def sgd(cost, params, lr=0.05, decay=1e-4):
	grads = T.grad(cost=cost, wrt=params)
	updates = []
	for p, g in zip(params, grads):
		updates.append([p, p - (g + decay*p) * lr])
	return updates

def sgd_momentum(cost, params, lr=0.05, decay=0.0001, momentum=0.5):
	grads = T.grad(cost=cost, wrt=params)
	updates = []
	for p, g in zip(params, grads):
		v = theano.shared(p.get_value()*0.)
		v_new = momentum*v - (g + decay*p) * lr 
		updates.append([p, p + v_new])
		updates.append([v, v_new])
	return updates

def RMSprop(cost, params, lr=0.001, decay=0.0001, rho=0.9, epsilon=1e-6):
	grads = T.grad(cost=cost, wrt=params)
	updates = []
	for p, g in zip(params, grads):
		acc = theano.shared(p.get_value() * 0.)
		acc_new = rho * acc + (1 - rho) * g ** 2
		gradient_scaling = T.sqrt(acc_new + epsilon)
		g = g / gradient_scaling
		updates.append((acc, acc_new))
		updates.append((p, p - lr * (g+ decay*p)))
	return updates

def testplot(conv1, pool1, conv2, pool2, mypath):
	plt.figure()
	plt.gray()
	plt.axis('off'); plt.imshow(teX[ind,:].reshape(28,28))
	plt.savefig(mypath+'input.png', bbox_inches='tight')
	plt.close()

	plt.figure()
	plt.gray()
	plt.title('1st convolved feature maps')
	for i in range(15):
		plt.subplot(5, 5, i+1); plt.axis('off'); plt.imshow(conv1[0,i,:].reshape(20,20))
	plt.savefig(mypath+'conv1.png', bbox_inches='tight')
	plt.close()

	plt.figure()
	plt.gray()
	plt.title('1st pooled feature maps')
	for i in range(15):
		plt.subplot(5, 5, i+1); plt.axis('off'); plt.imshow(pool1[0,i,:].reshape(10,10))
	plt.savefig(mypath+'pool1.png', bbox_inches='tight')
	plt.close()

	plt.figure()
	plt.gray()
	plt.title('2nd convolved feature maps')
	for i in range(20):
		plt.subplot(5, 5, i+1); plt.axis('off'); plt.imshow(conv2[0,i,:].reshape(6,6))
	plt.savefig(mypath+'conv2.png', bbox_inches='tight')
	plt.close()

	plt.figure()
	plt.gray()
	plt.title('2nd pooled feature maps')
	for i in range(20):
		plt.subplot(5, 5, i+1); plt.axis('off'); plt.imshow(pool2[0,i,:].reshape(3,3))
	plt.savefig(mypath+'pool2.png', bbox_inches='tight')
	plt.close()

def CAplot(epochs, a, trainning_cost, mypath):
	plt.figure()
	plt.plot(range(epochs), a)
	plt.xlabel('epochs')
	plt.ylabel('test accuracy')
	ymax = a.max()
	xmax = np.argmax(a)+1
	text= "Test accuracy max={:.3f}, epoch={:}".format(ymax, xmax)
	bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
	arrowprops=dict(arrowstyle="->",connectionstyle="arc3, rad=0")
	kw = dict(xycoords='data',textcoords="axes fraction",
			  arrowprops=arrowprops, bbox=bbox_props, ha="right", va="center")
	plt.gca().annotate(text, xy=(xmax,ymax), xytext=(0.94,0.6), **kw)
	plt.savefig(mypath+'test_accuracy.png')
	plt.close()

	plt.figure()
	plt.plot(range(epochs), trainning_cost)
	plt.xlabel('epochs')
	plt.ylabel('training cost')
	ymin = trainning_cost.min()
	xmin = np.argmin(trainning_cost)+1
	text= "Train cost min={:.3f}, epoch={:}".format(ymin, xmin)
	bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
	arrowprops=dict(arrowstyle="->",connectionstyle="arc3, rad=0")
	kw = dict(xycoords='data',textcoords="axes fraction",
			  arrowprops=arrowprops, bbox=bbox_props, ha="right", va="center")
	plt.gca().annotate(text, xy=(xmin,ymin), xytext=(0.94,0.6), **kw)
	plt.savefig(mypath+'training_cost.png')
	plt.close()

trX, teX, trY, teY = mnist(onehot=True)

trX = trX.reshape(-1, 1, 28, 28)
teX = teX.reshape(-1, 1, 28, 28)

trX, trY = trX[:12000], trY[:12000]
teX, teY = teX[:2000], teY[:2000]

X = T.tensor4('X')
Y = T.matrix('Y')

batch_size = 128
learning_rate = 0.05
decay = 1e-4

num_filters_1 = 15
num_filters_2 = 20
fcsize = 100
smsize = 10

w1, b1 = init_weights_bias4((num_filters_1, 1, 9, 9), X.dtype)
w2, b2 = init_weights_bias4((num_filters_2, num_filters_1, 5, 5), X.dtype)
w3, b3 = init_weights_bias2((num_filters_2*3*3, fcsize), X.dtype)
w4, b4 = init_weights_bias2((fcsize, smsize), X.dtype)

y1, o1, y2, o2, py_x = model(X, w1, b1, w2, b2, w3, b3, w4, b4)

y_x = T.argmax(py_x, axis=1)

cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y))
params = [w1, b1, w2, b2, w3, b3, w4, b4]

updates1 = sgd(cost, params, lr=0.05, decay=1e-4)
updates2 = sgd_momentum(cost, params, lr=0.05, decay=1e-4, momentum=0.1)
updates3 = RMSprop(cost, params, lr=0.001, decay=1e-4, rho=0.9, epsilon=1e-6)

train1 = theano.function(inputs=[X, Y], outputs=cost, updates=updates1, allow_input_downcast=True)
train2 = theano.function(inputs=[X, Y], outputs=cost, updates=updates2, allow_input_downcast=True)
train3 = theano.function(inputs=[X, Y], outputs=cost, updates=updates3, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)
test1 = theano.function(inputs = [X], outputs=[y1, o1], allow_input_downcast=True)
test2 = theano.function(inputs = [X], outputs=[y2, o2], allow_input_downcast=True)

print('sgd ..')
a = []
trainning_cost = []
for i in range(noIters):
	trX, trY = shuffle_data (trX, trY)
	cost = []
	for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
		cost.append(train1(trX[start:end], trY[start:end]))
	trainning_cost = np.append(trainning_cost, np.mean(cost))
	a = np.append(a, np.mean(np.argmax(teY, axis=1) == predict(teX)))
	print('No.%3d Accuracy = %f'%(i+1,a[i]))
ind = np.random.randint(low=0, high=2000)
convolved1, pooled1 = test1(teX[ind:ind+1,:])
convolved2, pooled2 = test2(teX[ind:ind+1,:])
testplot(convolved1, pooled1, convolved2, pooled2, mypath='2a_1/input1/')
ind = np.random.randint(low=0, high=2000)
convolved1, pooled1 = test1(teX[ind:ind+1,:])
convolved2, pooled2 = test2(teX[ind:ind+1,:])
testplot(convolved1, pooled1, convolved2, pooled2, mypath='2a_1/input2/')
CAplot(noIters, a, trainning_cost, mypath='2a_1/')
plt.figure(1)
plt.plot(range(noIters), a, label='sgd')
plt.figure(2)
plt.plot(range(noIters), trainning_cost, label='sgd')

print('sgd with momentum ..')
set_weights_bias4((num_filters_1, 1, 9, 9), X.dtype, w1, b1)
set_weights_bias4((num_filters_2, num_filters_1, 5, 5), X.dtype, w2, b2)
set_weights_bias2((num_filters_2*3*3, fcsize), X.dtype, w3, b3)
set_weights_bias2((fcsize, smsize), X.dtype, w4, b4)

a = []
trainning_cost = []
for i in range(noIters):
	trX, trY = shuffle_data (trX, trY)
	cost = []
	for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
		cost.append(train2(trX[start:end], trY[start:end]))
	trainning_cost = np.append(trainning_cost, np.mean(cost))
	a = np.append(a, np.mean(np.argmax(teY, axis=1) == predict(teX)))
	print('No.%3d Accuracy = %f'%(i+1,a[i]))

ind = np.random.randint(low=0, high=2000)
convolved1, pooled1 = test1(teX[ind:ind+1,:])
convolved2, pooled2 = test2(teX[ind:ind+1,:])
testplot(convolved1, pooled1, convolved2, pooled2, mypath='2a_2/input1/')
ind = np.random.randint(low=0, high=2000)
convolved1, pooled1 = test1(teX[ind:ind+1,:])
convolved2, pooled2 = test2(teX[ind:ind+1,:])
testplot(convolved1, pooled1, convolved2, pooled2, mypath='2a_2/input2/')
CAplot(noIters, a, trainning_cost, mypath='2a_2/')
plt.figure(1)
plt.plot(range(noIters), a, label='sgd with momentum')
plt.figure(2)
plt.plot(range(noIters), trainning_cost, label='sgd with momentum')

print('RMSprop ..')
set_weights_bias4((num_filters_1, 1, 9, 9), X.dtype, w1, b1)
set_weights_bias4((num_filters_2, num_filters_1, 5, 5), X.dtype, w2, b2)
set_weights_bias2((num_filters_2*3*3, fcsize), X.dtype, w3, b3)
set_weights_bias2((fcsize, smsize), X.dtype, w4, b4)

a = []
trainning_cost = []
for i in range(noIters):
	trX, trY = shuffle_data (trX, trY)
	cost = []
	for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
		cost.append(train3(trX[start:end], trY[start:end]))
	trainning_cost = np.append(trainning_cost, np.mean(cost))
	a = np.append(a, np.mean(np.argmax(teY, axis=1) == predict(teX)))
	print('No.%3d Accuracy = %f'%(i+1,a[i]))
ind = np.random.randint(low=0, high=2000)
convolved1, pooled1 = test1(teX[ind:ind+1,:])
convolved2, pooled2 = test2(teX[ind:ind+1,:])
testplot(convolved1, pooled1, convolved2, pooled2, mypath='2a_3/input1/')
ind = np.random.randint(low=0, high=2000)
convolved1, pooled1 = test1(teX[ind:ind+1,:])
convolved2, pooled2 = test2(teX[ind:ind+1,:])
testplot(convolved1, pooled1, convolved2, pooled2, mypath='2a_3/input2/')
CAplot(noIters, a, trainning_cost, mypath='2a_3/')
plt.figure(1)
plt.plot(range(noIters), a, label='RMSProp')
plt.xlabel('epochs')
plt.ylabel('test accuracy')
plt.legend(loc='lower right')
plt.savefig('testAccuracy.png', bbox_inches='tight')
plt.figure(2)
plt.plot(range(noIters), trainning_cost, label='RMSProp')
plt.xlabel('epochs')
plt.ylabel('Training cost')
plt.legend(loc='upper right')
plt.savefig('trainningCost.png', bbox_inches='tight')