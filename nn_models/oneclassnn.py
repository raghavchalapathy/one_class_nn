from data_fetch import prepare_usps_mlfetch
[Xtrue,Xlabels] = prepare_usps_mlfetch()
data = Xtrue

# series, not a new dataframe
target = Xlabels
data_train = data[0:220]
targets_train = target[0:220]
data_test = data[220:231]
targets_test = target[220:231]


from tflearn import DNN
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression, oneClassNN
from tflearn.metrics import binary_accuracy_op
from time import time
from sklearn import metrics
import matplotlib.pyplot as plt
import tensorflow as tf
import tflearn
import numpy as np
# Clear all the graph variables created in previous run and start fresh
tf.reset_default_graph()
init = tf.global_variables_initializer()
# Training examples
X = data_train
# Y = [[0], [0], [0], [0]]
Y = targets_train
# Y = list(Y)
Y = Y.tolist()
Y = [[i] for i in Y]

# For testing the algorithm
X_test = data_test
Y_test = targets_test
Y_test = Y_test.tolist()
Y_test = [[i] for i in Y_test]

m, n = data_train.shape
No_of_inputNodes = n
No_of_hiddenNodes = n
print "No_of_hiddenNodes", No_of_hiddenNodes

# Histogram of train, test
# AUC: computation test;

input_layer = input_data(shape=[None, No_of_inputNodes])  # input layer of size 2
# hidden_layer = fully_connected(input_layer, 4, bias=False, activation='sigmoid', name="hiddenLayer_Weights",
#                                weights_init="normal")  # hidden layer of size 2
#
#
# output_layer = fully_connected(hidden_layer, 1, bias=False, activation='linear', name="outputLayer_Weights",
#                                weights_init="normal")  # output layer of size 1

hidden_layer1 = fully_connected(input_layer, 16, bias=False, activation='sigmoid', name="hiddenLayer_Weights",
                               weights_init="normal")  # hidden layer of size 2

hidden_layer2 = fully_connected(hidden_layer1, 4, bias=False, activation='sigmoid', name="hiddenLayer_Weights",
                               weights_init="normal")  # hidden layer of size 2
output_layer = fully_connected(hidden_layer2, 1, bias=False, activation='linear', name="outputLayer_Weights",
                               weights_init="normal")  # output layer of size 1



## Initialize the weights with random.normal and seed= 42
# hidden_layer.W = tflearn.initializations.normal(mean=0.0, stddev=1.0, dtype=tf.float32, seed=42)
# output_layer.W = tflearn.initializations.normal(mean=0.0, stddev=1.0, dtype=tf.float32, seed=42)

# assign the learnt weights
# wStar = hidden_layer.W
# VStar = output_layer.W

wStar = hidden_layer2.W
VStar = output_layer.W

# Hyper parameters for the one class Neural Network
v = 0.04
import tflearn.variables as va

value = 0.5
init = tf.constant_initializer(value)
rho = va.variable(name='rho', dtype=tf.float32, shape=[], initializer=init)
# rho_previous = va.variable(name='rho_previous',dtype=tf.float32,shape=[] )
# rho_next = va.variable(name='rho_next', dtype=tf.float32,shape=[])
rcomputed = []
auc = []
# rho=0.3
# oneClassNN = oneClassNN(output_layer, v, rho, hidden_layer, output_layer, optimizer='sgd', loss='OneClassNN_Loss',
#                         learning_rate=5)

oneClassNN = oneClassNN(output_layer, v, rho, hidden_layer2, output_layer, optimizer='sgd', loss='OneClassNN_Loss',
                        learning_rate=5)
model = DNN(oneClassNN, tensorboard_verbose=3)
model.fit(X, Y, n_epoch=200, show_metric=True)
y_pred = model.predict(data_train)  # Apply some ops
rho = np.percentile(y_pred, v * 100)
rcomputed.append(rho)
# # Define the Iteration for optimising and stabilizing the value of r
iterStep = 0

while (iterStep <= 1000):
    print "Running Iteration :", iterStep
    y_pred = model.predict(data_train)  # Apply some ops
    #     y_pred = np.sort(y_pred) # Sort in ascending order
    #     rhoIndex = int(v * len(data_train))
    #   rho=(y_pred[rhoIndex] + y_pred[rhoIndex+1])/2
    rho = np.percentile(y_pred, v * 100)
    rStar = rho

    #     rho_next =rho
    model.fit(X, Y, n_epoch=1, show_metric=True, batch_size=220)
    iterStep = iterStep + 1
    rcomputed.append(rho)
    print "rho", rho

# The data_train and data_test
data_train = data[0:220]
y_predTrain = model.predict(data_train)  # Apply some ops

data_test = data[220:231]
y_predTest = model.predict(data_test)

## PLot the AUC for the data
# from sklearn.cross_validation import train_test_split
# train_data, test_data, train_target, test_target = train_test_split(data, target, train_size = 0.8)
# preds = model.predict(test_data)
# targs = test_target

# oneClass_nn_score = metrics.accuracy_score(targs, preds.round())
# # Compute the AUC for OneClassNN
# fpr, tpr, thresholds = metrics.roc_curve(targs, preds)
# OCNN_auc_score = metrics.auc(fpr, tpr)
# auc.append(OCNN_auc_score)

print type(wStar)
print type(VStar)
# # print "Value of R computed is... ",rcomputed
# # Make a figure
# fig1 = plt.figure()

# plt.hist(y_predTrain,  alpha=0.5, label='normal')
# plt.hist(y_predTest,  alpha=0.5, label='anomalies')
# plt.legend(loc='upper right')
# plt.xlabel("Samples")
# plt.ylabel("Y_hat(Predicted Value)")
# plt.title("OneClass-NN(Output=Y_hat) Vs Samples")
# plt.show()

# # Plot lines 1-3
# fig2 = plt.figure()
# line1 = plt.plot(y_predTrain-rcomputed[0],'bo-',label='Normal')
# line2 = plt.plot(y_predTest-rcomputed[0],'go-',label='Anomalies')
# plt.title("Anomaly-Score(Yhat-r)")
# plt.xlabel("Samples")
# plt.ylabel("Score")
# plt.legend(loc='lower right')
# # Display the figure
# plt.show()

# # AUC Score computed for the data points
# fig3 = plt.figure()
# line1 = plt.plot(auc,'ro-',label='OneClass-NN')
# plt.title("AUC Curve")
# # Display the figure
# plt.show()

print type(wStar)
print type(VStar)
print wStar.shape
print VStar.shape
print data_train.shape
print wStar.dtype
print VStar.dtype
import tflearn

X = X.astype(np.float32)
X_test = data_test
X_test = X_test.astype(np.float32)
print type(X)
print X.dtype
g = lambda x: x


def nnScore(X, w, V, g):
    return tf.matmul(g((tf.matmul(X, w))), V)


nu = 0.04

import tensorflow as tf

# k = tf.placeholder(tf.float32, shape=(None, 256))
# Make a normal distribution, with a shifting mean
train = nnScore(X, wStar, VStar, g)
test = nnScore(X_test, wStar, VStar, g)
# # Record that distribution into a histogram summary
# tf.summary.histogram('r_____%1.6f' % rStar, train)
# tf.summary.histogram('r_____%1.6f' % rStar, test)

# # Setup a session and summary writer

# writer = tf.summary.FileWriter("/tmp/histogram_example")

# summaries = tf.summary.merge_all()
sess = tf.Session()
print (train.shape)
print (test.shape)
print train
print test
# tflearn.helpers.summarizer.summarize_variables (train_vars=train, summary_collection='tflearn_summ')
# tflearn.helpers.summarizer.summarize_variables (train_vars=test, summary_collection='tflearn_summ')
sess.run(tf.initialize_all_variables())
# plt.xlim(-4e-8,8e-8)
arrayTrain = train.eval(session=sess)
arrayTest = test.eval(session=sess)
plt.hist(arrayTrain / rStar, label='Normal');
plt.hist(arrayTest / rStar, label='Anomalies');
plt.legend(loc='upper right')
plt.title('r = %1.6f' % rStar)
plt.show()

plt.figure(figsize=(10,8))
plt.title('R-Value', fontsize=20)
plt.xlabel('n-epochs', fontsize=15)
plt.ylabel('r', fontsize=15)
plt.plot(rcomputed, color='b')
plt.show()

# Setup a loop and write the summaries to disk
# N = 1
# for step in range(N):
#   k_val = step/float(N)
#   summ = sess.run(summaries,feed_dict={k: k_val})
#   writer.add_summary(summ, global_step=step)



# print(rStar)
# result = nnScore(X, wStar, VStar, g)
# print type(result)
# print result.shape
# res1 = va.variable(name='res2', dtype=tf.float32,shape=[220,1])
# with sess.as_default():
#     tflearn.variables.set_value(res1, result)
# print(np.percentile(nnScore(X, wStar, VStar, g), q = 100 * nu))
