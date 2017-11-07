import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import time

def init_bias(n = 1):
	return(theano.shared(np.zeros(n), theano.config.floatX))

def init_weights(n_in=1, n_out=1, logistic=True):
	W_values = np.asarray(
		np.random.uniform(
		low=-np.sqrt(6. / (n_in + n_out)),
		high=np.sqrt(6. / (n_in + n_out)),
		size=(n_in, n_out)),
		dtype=theano.config.floatX
		)
	if logistic == True:
		W_values *= 4
	return (theano.shared(value=W_values, name='W', borrow=True))

# scale data
def scale(X, X_min, X_max):
	return (X - X_min)/(X_max-X_min)

# update parameters
def sgd(cost, params, lr=0.01):
	grads = T.grad(cost=cost, wrt=params)
	updates = []
	for p, g in zip(params, grads):
		updates.append([p, p - g * lr])
	return updates

def shuffle_data (samples, labels):
	idx = np.arange(samples.shape[0])
	np.random.shuffle(idx)
	samples, labels = samples[idx], labels[idx]
	return samples, labels

learning_rate = 0.01
epochs = 1000

def initialize(hnno, decay):
	global X, Y, w1, b1, w2, b2, h1, py, y_x, cost, params, updates, train, predict
	# theano expressions
	X = T.matrix() #features
	Y = T.matrix() #output

	w1, b1 = init_weights(36, hnno), init_bias(hnno) #weights and biases from input to hidden layer
	w2, b2 = init_weights(hnno, 6, logistic=False), init_bias(6) #weights and biases from hidden to output layer

	h1 = T.nnet.sigmoid(T.dot(X, w1) + b1)
	py = T.nnet.softmax(T.dot(h1, w2) + b2)

	y_x = T.argmax(py, axis=1)

	cost = T.mean(T.nnet.categorical_crossentropy(py, Y)) + decay*(T.sum(T.sqr(w1)+T.sum(T.sqr(w2))))
	params = [w1, b1, w2, b2]
	updates = sgd(cost, params, learning_rate)

	# compile
	train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
	predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

#read train data
train_input = np.loadtxt('sat_train.txt',delimiter=' ')
trainX, train_Y = train_input[:,:36], train_input[:,-1].astype(int)
trainX_min, trainX_max = np.min(trainX, axis=0), np.max(trainX, axis=0)
trainX = scale(trainX, trainX_min, trainX_max)

train_Y[train_Y == 7] = 6
trainY = np.zeros((train_Y.shape[0], 6))
trainY[np.arange(train_Y.shape[0]), train_Y-1] = 1

#read test data
test_input = np.loadtxt('sat_test.txt',delimiter=' ')
testX, test_Y = test_input[:,:36], test_input[:,-1].astype(int)

testX_min, testX_max = np.min(testX, axis=0), np.max(testX, axis=0)
testX = scale(testX, testX_min, testX_max)

test_Y[test_Y == 7] = 6
testY = np.zeros((test_Y.shape[0], 6))
testY[np.arange(test_Y.shape[0]), test_Y-1] = 1

print(trainX.shape, trainY.shape)
print(testX.shape, testY.shape)

#Plots
def plots(opt):
	global epochs, train_cost, test_accuracy
	opt = str(opt)
	plt.figure()
	plt.plot(range(epochs), train_cost*100)
	plt.xlabel('Iterations')
	plt.ylabel('Cross-entropy')
	ymin = 100*train_cost.min()
	xmin = np.argmin(train_cost)+1
	text= "Train_cost_min={:.3f}, epoch={:}".format(ymin, xmin)
	bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
	arrowprops=dict(arrowstyle="->",connectionstyle="arc3, rad=0")
	kw = dict(xycoords='data',textcoords="axes fraction",
			  arrowprops=arrowprops, bbox=bbox_props, ha="right", va="center")
	plt.gca().annotate(text, xy=(xmin,ymin), xytext=(0.94,0.6), **kw)
	plt.title('training cost with decay='+opt)
	plt.savefig('p1a_cost_'+opt+'.png')

	plt.figure()
	plt.plot(range(epochs), test_accuracy*100)
	plt.xlabel('Iterations')
	plt.ylabel('Accuracy')
	ymax = 100*test_accuracy.max()
	xmax = np.argmax(test_accuracy)+1
	text= "test_accuracy_max={:.3f}, epoch={:}".format(ymax, xmax)
	bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
	arrowprops=dict(arrowstyle="->",connectionstyle="arc3, rad=0")
	kw = dict(xycoords='data',textcoords="axes fraction",
			  arrowprops=arrowprops, bbox=bbox_props, ha="right", va="center")
	plt.gca().annotate(text, xy=(xmax,ymax), xytext=(0.94,0.6), **kw)
	plt.title('test accuracy with decay='+opt)
	plt.savefig('p1a_accuracy_'+opt+'.png')

# train and test
def tnt(batch_size, hnno, decay, trainX, trainY):
	initialize(hnno, decay)
	global test_accuracy, train_cost, epochs, times, Accuracy_plot
	t1 = time.time()
	n = len(trainX)
	for i in range(epochs):
		trainX, trainY = shuffle_data(trainX, trainY)
		cost = 0.0
		for start, end in zip(range(0, n, batch_size), range(batch_size, n, batch_size)):
			cost += train(trainX[start:end], trainY[start:end])
		train_cost = np.append(train_cost, cost/(n // batch_size))
		test_accuracy = np.append(test_accuracy, np.mean(np.argmax(testY, axis=1) == predict(testX)))
	times.append(time.time()-t1)
	print('Time: %.2fs'%times[-1])
	accuracy = (np.max(test_accuracy)*100, np.argmax(test_accuracy)+1)
	Accuracy_plot.append(accuracy[0])
	print('%.2f accuracy at %d iterations'%accuracy)

def decayplot():
	plt.figure()
	xtick = [0,1,2,3,4]
	xdecay = ['1e-3','1e-6','1e-9','1e-12','0']
	plt.xticks(xtick,xdecay)
	plt.plot(xtick, Accuracy_plot)
	plt.xlabel('Decay')
	plt.ylabel('Accuracy')
	plt.title('Decay against Accuracy')
	plt.savefig('p1a_decay_accuracy.png')

decayA = [1e-3,1e-6,1e-9,1e-12,0]
times = []
Accuracy_plot = []

for decay in decayA:
	batch_size = 4
	hnno = 10
	test_accuracy = []
	train_cost = []
	print('Decay parameter = '+str(decay)+' running')
	tnt(batch_size, hnno, decay, trainX, trainY)
	plots(decay)
decayplot()