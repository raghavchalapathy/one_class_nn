from tflearn import DNN
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression, oneClassNN
import tensorflow as tf
import tflearn
import numpy as np
import tflearn.variables as va
import numpy  as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as srn

from data_fetch import prepare_usps_mlfetch

[Xtrue,Xlabels] = prepare_usps_mlfetch()

data  = Xtrue
label = Xlabels
data_train    = data[0:220]
data_test     = data[220:231]
targets_train = label[0:220]
targets_test  = label[220:231]
# Clear all the graph variables created in previous run and start fresh
# tf.reset_default_graph()

## Set up the data for running the algorithm
data_train = data[0:220]
target = Xlabels
X = data_train
Y = targets_train
Y = Y.tolist()
Y = [[i] for i in Y]
data_train = data[0:220]
targets_train = target[0:220]
data_test = data[220:231]
targets_test = target[220:231]
# For testing the algorithm
X_test = data_test
Y_test = targets_test
Y_test = Y_test.tolist()
Y_test = [[i] for i in Y_test]

No_of_inputNodes = X.shape[1]
K = 4
nu = 0.04

# Define the network
input_layer = input_data(shape=[None, No_of_inputNodes])  # input layer of size
hidden_layer = fully_connected(input_layer, 4, bias=False, activation='sigmoid', name="hiddenLayer_Weights",
                               weights_init="normal")  # hidden layer of size 2
output_layer = fully_connected(hidden_layer, 1, bias=False, activation='linear', name="outputLayer_Weights",
                               weights_init="normal")  # output layer of size 1

# assign the learnt weights
wStar = hidden_layer.W
VStar = output_layer.W

# Hyper parameters for the one class Neural Network
v = 0.04
nu = 0.04

# Initialize rho
value = 0.01
init = tf.constant_initializer(value)
rho = va.variable(name='rho', dtype=tf.float32, shape=[], initializer=init)
rcomputed = []
auc = []


sess = tf.Session()
sess.run(tf.initialize_all_variables())
print sess.run(tflearn.get_training_mode()) #False
tflearn.is_training(True, session=sess)
print sess.run(tflearn.get_training_mode())  #now True

oneClassNN = oneClassNN(output_layer, v, rho, hidden_layer, output_layer, optimizer='sgd', loss='OneClassNN_Loss',
                            learning_rate=5)

model = DNN(oneClassNN, tensorboard_verbose=3)

iterStep = 0
while (iterStep < 100):
    print "Running Iteration :", iterStep
    # Call the cost function


    y_pred = model.predict(data_train)  # Apply some ops
    y_pred_test = model.predict(data_test)  # Apply some ops
    value = np.percentile(y_pred, v * 100)
    tflearn.variables.set_value(rho, value,session=sess)
    rStar = rho
    model.fit(X, Y, n_epoch=1, show_metric=True, batch_size=220)
    iterStep = iterStep + 1
    rcomputed.append(rho)
    temp = tflearn.variables.get_value(rho, session=sess)


    print "y_pred",y_pred
    print "y_predTest", y_pred_test

g = lambda x: x


def nnScore(X, w, V, g):
    return tf.matmul(g((tf.matmul(X, w))), V)


# Format the datatype to suite the computation of nnscore
X = X.astype(np.float32)
X_test = data_test
X_test = X_test.astype(np.float32)

train = nnScore(X, wStar, VStar, g)
test = nnScore(X_test, wStar, VStar, g)

# Access the value inside the train and test for plotting
# Create a new session and run the example
# sess = tf.Session()
# sess.run(tf.initialize_all_variables())
arrayTrain = train.eval(session=sess)
arrayTest = test.eval(session=sess)

print "Train Array:",arrayTrain
print "Test Array:",arrayTest

plt.hist(arrayTrain, label='Normal');
plt.hist(arrayTest, label='Anomalies');
plt.legend(loc='upper right')
# plt.title('r = %1.6f' % rStar)
plt.show()
