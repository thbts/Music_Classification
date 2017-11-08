import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, BatchNormalization, Conv1D, Flatten, MaxPooling1D
from keras import optimizers
from keras.optimizers import Nadam
import keras.utils
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import os
import time
from subprocess import check_output
from datetime import datetime

conv = True
PCA_act = False
features_pca = 100

# set the necessary directories
train_data = pd.read_csv("data/train_data.csv",header=None)
train_labels = pd.read_csv("data/train_labels.csv",header=None)
test_data = pd.read_csv("data/test_data.csv",header=None)

train_data = np.array(train_data)
train_labels = np.array(train_labels)
test_data = np.array(test_data)

train_data = preprocessing.scale(train_data)
test_data = preprocessing.scale(test_data)

train_labels_acc = train_labels
for i in range(0,len(train_labels)):
    train_labels[i]=train_labels[i]-1
train_labels_acc = train_labels
train_labels = keras.utils.to_categorical(train_labels, num_classes=10)
    

samples = int(train_data.shape[0])
features = int(train_data.shape[1])


## PCA to reduce the dimensionality of the data
if(PCA_act):
	sklearn_pca = PCA(n_components=features_pca)
	sklearn_pca.fit(train_data)
	train_data = sklearn_pca.transform(train_data)
	test_data = sklearn_pca.transform(test_data)

if(conv):
	train_data.shape = (train_data.shape[0],train_data.shape[1],1)
	test_data.shape = (test_data.shape[0],test_data.shape[1],1)


useful_features = train_data.shape[1]

# First let's define the two different types of layers that we will be using.
def neural_network():
	model = Sequential()
	model.add(Dense(20, activation='relu', input_shape=(useful_features,)))
	model.add(BatchNormalization())
	model.add(Dropout(0.3))
	model.add(Dense(10, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.3))
	model.add(Dense(10, activation='softmax'))
	model.compile(optimizer=Nadam(), loss='categorical_crossentropy',metrics=['accuracy'])
	return model

def neural_network_conv():
	model = Sequential()
	model.add(Conv1D(20,6, activation='relu', input_shape=(useful_features,1)))
	model.add(BatchNormalization())
	model.add(Dropout(0.25))
	model.add(Conv1D(10,3, activation='relu', input_shape=(useful_features,1)))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))
	model.add(MaxPooling1D(2))
	model.add(Conv1D(10,2, activation='relu', input_shape=(useful_features,1)))
	model.add(BatchNormalization())
	model.add(Dropout(0.25))
	model.add(Conv1D(1,2, activation='relu', input_shape=(useful_features,1)))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))
	model.add(MaxPooling1D(2))
	#model.add(MaxPooling1D(pool_size=2, strides=None, padding='valid'))
	model.add(Flatten())
	model.add(Dense(10, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.3))
	model.add(Dense(10, activation='softmax'))
	model.compile(optimizer=Nadam(), loss='categorical_crossentropy',metrics=['accuracy'])
	return model

def inverse_to_categorical(array):
	res = []
	for i in range(0,len(array)):
		argmax = np.argmax(array[i])
		res.append(argmax)
	return res

def train_model(model, trainings, train_data,train_labels,validation_data,validation_labels):
	for i in range(0,trainings):
		print('Training n°'+str(i))
		train_data,train_labels = shuffle(train_data,train_labels)
		model.fit(train_data,train_labels,epochs=10,batch_size=16, verbose = 2)
		print("accuracy of neural network :")
		print(model.evaluate(validation_data,validation_labels))


#Validation of the model
errors = []

for K in range(0,5):
	#Pick which songs we will use for testing and for validation
	train1_data, validation_data, train1_labels, validation_labels = train_test_split(train_data, train_labels, train_size=0.8, test_size=0.2)
	
	#Training time
	if(conv):
		model = neural_network_conv()
	else:
		model = neural_network()
	train_model(model,10,train1_data,train1_labels,validation_data,validation_labels)
	
	err = model.evaluate(validation_data,validation_labels)
	print("End of training. Final accuracy : ")
	print(err)
	errors.append(err)

errors = np.array(errors)


# training on the whole dataset for the submission
if(conv):
	model = neural_network_conv()
else:
	model = neural_network()

for trains in range(0,10):
	print('Training n°'+str(trains))
	train_data,train_labels = shuffle(train_data,train_labels)
	model.fit(train_data, train_labels, epochs=10,batch_size=16, verbose = 2)

	

## Submission
y_pred = model.predict(test_data)
y_pred_accuracy = []
index = []
for i in range(0,len(y_pred)):
    argmax = np.argmax(y_pred[i])+1
    y_pred_accuracy.append(argmax)
    index.append(i+1)


output = pd.DataFrame({'Sample_id': index,
        'Sample_label': y_pred_accuracy})


print( "\nWriting results to disk:" )
output.to_csv('ANN_{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)

print( "\nFinished!" )


