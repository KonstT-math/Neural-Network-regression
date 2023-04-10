import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

dev_length = 10
nofeats = 8

# load dataset
dataframe = pd.read_csv("concrete.csv")
dataset = dataframe.values

# split into input (X) and output (Y) variables
X = dataset[:,0: nofeats]
Y = dataset[:,nofeats]

# model: one fully connected hidden layer. ReLU activation function for the hidden layer. No activation function is used for the output layer because it is a regression problem.

# define base model
def baseline_model():
 # create model
 model = Sequential()
 model.add(Dense(13, input_shape=(nofeats,), kernel_initializer='normal', activation='relu'))
 model.add(Dense(1, kernel_initializer='normal'))
 # Compile model
 model.compile(loss='mean_squared_error', optimizer='adam')
 return model

estimator = KerasRegressor(model=baseline_model, epochs=100, batch_size=5, verbose=0)

kfold = KFold(n_splits=10)
results = cross_val_score(estimator, X, Y, cv=kfold, scoring='neg_mean_squared_error')
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

data_dev = dataset[0:dev_length]
X_dev = data_dev[:,0:nofeats]
Y_dev = data_dev[:,nofeats]

print("--------------")
print("--------------")
estimator.fit(X, Y)

difference = []
for i in range(10):
	pred = X_dev[i].reshape(1,-1)
	#prediction.append(estimator.predict(pred))
	print(estimator.predict(pred) , Y_dev[i])
	difference.append(np.abs(estimator.predict(pred) - Y_dev[i]))

#print("--------------")
#print("Accuracy = ", difference)
print("--------------")
print("Accuracy = ", np.mean(difference))
print("--------------")






