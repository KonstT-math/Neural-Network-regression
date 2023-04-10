import pandas as pd
import numpy as np

# -----------------------------------------

def select_data(data, ny):
	
	yName = data.columns[ny]
	Y = data[yName]

	return Y

# constants
# nodes of hidden layer
nod1 = 13
# number of test data
dev_length = 10
# learning rate
learning_rate = 0.15
# number of epochs
noepochs = 10000
# number of features
nofeats = 8

np.random.seed(1)

# -----------------------------------------

# data:
data1 = pd.read_csv('concrete.csv')
#data = pd.read_csv('con_cube.csv')
data = pd.read_csv('con_std.csv')
data = np.array(data)
m,n = data.shape

#np.random.shuffle(data)

data_dev = data[0:dev_length]
X_dev = data_dev[:,0:nofeats]
Y_dev = data_dev[:,nofeats]
Y_dev = Y_dev.reshape(1,-1)
Y_dev = Y_dev.T

data_train = data[dev_length:m]
X_train = data_train[:, 0:nofeats] 
Y_train = data_train[:,nofeats] 
Y_train = Y_train.reshape(1,-1)
Y_train = Y_train.T

# -----------------------------------------

def mse(y_true, y_pred):
	return np.mean(np.power(y_true-y_pred, 2));

def ReLU(Z):
	return np.maximum(0,Z)

def d_ReLU(Z):
	return Z>0

def tanh(x):
    return np.tanh(x);

def d_tanh(x):
    return 1-np.tanh(x)**2;

def sig(x):
    return 1/(1 + np.exp(-x))

def d_sig(x):
    return sig(x)*(1- sig(x))

def init_weights():
	W1 = np.random.rand(nofeats,nod1)-0.5 
	b1 = np.random.rand(1,nod1)-0.5 

	W2 = np.random.rand(nod1,1)-0.5 
	b2 = np.random.rand(1,1)-0.5 
	return W1, b1, W2, b2

def forward_prop(W1, b1, W2, b2, X):
	Z1 = np.dot(X,W1) + b1 
	A1 = tanh(Z1)
	Z2 = np.dot(A1,W2) +b2
	A2 = Z2
	return Z1, A1, Z2, A2

def back_prop(Z1, A1, Z2, A2, W2, X, Y):
	m = Y.size
	dZ2 = (A2 - Y) 
	dW2 = (1/m)*np.dot(A1.T, dZ2)
	db2 = np.mean(dZ2,axis=0, keepdims=True) 

	dZ1 = np.dot(dZ2, W2.T) * d_tanh(A1) 
	dW1 = (1/m)*np.dot(X.T, dZ1) 
	db1 = np.mean(dZ1, axis=0, keepdims=True)

	return dW1, db1, dW2, db2


def update_weights(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
	W1 = W1 - alpha*dW1 
	b1 = b1 - alpha*db1 
	W2 = W2 - alpha*dW2 
	b2 = b2 - alpha*db2 
	return W1, b1, W2, b2

def nn(X, Y, epochs, alpha):
	W1, b1, W2, b2 = init_weights()	
	for i in range(epochs):
		# forward propagation
		Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
		loss = mse(A2, Y)
		if i%100 == 0:
			print("loss at iteration" , i, loss)		
		# backward propagation
		dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X, Y)
		W1, b1, W2, b2 = update_weights(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
		
	return W1, b1, W2, b2


W1, b1, W2, b2 = nn(X_train, Y_train, noepochs, learning_rate)

# create csv files with current weights and biases
dfW1 = pd.DataFrame(W1)
dfW1.to_csv(r'L1weight1.csv', index = False, header=True)

dfb1 = pd.DataFrame(b1)
dfb1.to_csv(r'L1bias1.csv', index = False, header=True)

dfW2 = pd.DataFrame(W2)
dfW2.to_csv(r'L1weight2.csv', index = False, header=True)

dfb2 = pd.DataFrame(b2)
dfb2.to_csv(r'L1bias2.csv', index = False, header=True)



print("------")

meano = np.mean(select_data(data1,nofeats))
stdev = np.std(select_data(data1,nofeats))

difference = []
for i in range(dev_length):
	a1 = tanh(np.dot( X_dev[i] ,W1 ) +b1)
	a2 = np.dot(a1,W2) +b2
	out1 = a2
	print("decision:")
	print(out1, Y_dev[i])
	#print(np.power(out1,3), np.power(Y_dev[i],3))
	
	#out_real = Y_dev[i]
	#out_predict = out1
	out_real = Y_dev[i]*stdev + meano
	out_predict = out1*stdev + meano
	#out_real = np.power(Y_dev[i],3)
	#out_predict = np.power(out1,3)
	print(out_predict, out_real)
	difference.append(np.abs(out_predict - out_real))

print("------")
print("accuracy (mean difference) = ", np.mean(difference))



