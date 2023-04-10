# Neural-Network-regression
Neural Network from scratch. Preprocess is used on data.

preprocess.py file runs KS-test of normality on each feature from dataframe. Also, rescales dataset as needed (standard transformation, log transformation, square root transformation, cube root transformation). After appropriate rescaling is applied, a new csv file is created for use in regression.

regression_1L.py is a fully connected neural network with 1 hidden layer for regression, written in pure python.

regression_2L.py is a fully connected neural network with 2 hidden layers for regression, written in pure python.

nn_tf.py is a regression neural network utilizing tensorflow/keras. This file is a testing ground for the former code; by adjusting parameters/architecture/preprocessing in regression_1L.py and regression_2L.py, one can compare results with the standard code in nn_tf.py

