import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.losses import BinaryCrossentropy
import os
from sklearn.model_selection import train_test_split
# Beispiele aus Internet, Daten laden und Datensatz erstellen muss noch auf unseren Anwendungsfall angepasst werden

#Hyperparameters
epochs = 3
batch_size = 32

"""
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

numpy.random.seed(7)
# load the dataset
dataframe = read_csv('airline-passengers.csv', usecols=[1], engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

train, test = ...
# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))


model = Sequential()
model.add(LSTM(128, input_shape=(x_train.shape[1:]), activation = 'relu', return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(128, activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(32, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation = 'softmax'))

loss_function = BinaryCrossentropy()

model.compile(loss=loss_function, optimizer='adam', metrics=['accuracy'])
model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size)
"""
def lstm(data, target):

	data = np.concatenate(data, axis=0 )
	target = np.concatenate(target, axis=0 )

	X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

	print(X_train.shape)
	print(y_train.shape)
	print(X_train)
	print(y_train)

	
	model = Sequential()
	model.add(LSTM(32, input_shape=(X_train.shape), activation = 'relu', return_sequences = True))
	model.add(Dropout(0.2))
	#model.add(LSTM(128, activation = 'relu'))
	#model.add(Dropout(0.1))
	#model.add(Dense(32, activation = 'relu'))
	#model.add(Dropout(0.2))
	model.add(Dense(2, activation = 'softmax'))

	loss_function = BinaryCrossentropy()

	model.compile(loss=loss_function, optimizer='adam', metrics=['accuracy'])
	model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
	
	return


"""
breath_data = read_csv("C:/Users/inesg/Documents/Uni/Master/Seminar/breath_data/190114_peep_tit_1.csv")
compliance_data = breath_data.iloc[:, 16:17].values

print(np.shape(compliance_data))

scaler = MinMaxScaler(feature_range = (0, 1))
compliance_scaled = scaler.fit_transform(compliance_data)

features_set = []
labels = []
for i in range(600, 1818):
	features_set.append(compliance_scaled[i-600:i, 0])
	labels.append(compliance_scaled[i, 0])

features_set, labels = np.array(features_set), np.array(labels)
features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(features_set.shape[1], 1)))
model.add(Dropout(0.2))
model.add(Dense(units = 1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.fit(features_set, labels, epochs = 3, batch_size = 32)


test_data = read_csv("C:/Users/inesg/Documents/Uni/Master/Seminar/breath_data/190116_peep_tit_1.csv")
compliance_testing = test_data.iloc[:, 16:17].values
apple_total = pd.concat((breath_data['C_rs_est'], test_data['C_rs_est']), axis=0)
test_inputs = apple_total[len(apple_total) - len(test_data) - 600:].values

test_inputs = test_inputs.reshape(-1,1)
test_inputs = scaler.transform(test_inputs)

test_features = []
for i in range(600, 1870):
    test_features.append(test_inputs[i-600:i, 0])
test_features = np.array(test_features)
test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], 1))
predictions = model.predict(test_features)
predictions = scaler.inverse_transform(predictions)

plt.figure(figsize=(10,6))
plt.plot(compliance_testing, color='blue', label='Actual Apple Stock Price')
plt.plot(predictions , color='red', label='Predicted Apple Stock Price')
plt.title('Apple Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Apple Stock Price')
plt.legend()
plt.show()
"""

"""
def create_data(set, name):
	data_set = set[["p_peep", "C_rs_est"]].to_numpy()
	width = 2
	length = 20
	peep = 23
	list = []
	while peep >= 5:
		data = data_set[(data_set[..., 0] > peep) & (data_set[..., 0] < (peep+2))].astype(np.float16)
		array = np.empty([length, width], np.float16)
		for x in range(0, length):
			array[x, :] = data[x, :]
		list.append(array)
		peep = peep-2
	
	target = target_vector(list, length, width)
	data = np.concatenate((list[0], list[1], list[2], list[3], list[4], list[5], list[6], list[7], list[8], list[9]))

	return data[:,1:4], target


def target_vector(list, length, width):

    target_list = []
    for x in range(0,10):
        average = np.average(list[x][..., -1])
        target_list.append(average)

    max_value = max(target_list)
    max_index = target_list.index(max_value)

    target = np.zeros((length*10, 1))
    target[(max_index*length):(max_index*length+length)] = np.ones((length, 1))

    return target
		

data_list = []
target_list = []
filenames = []
all_filenames = os.listdir('breath_data_optimal')
for filename in all_filenames:
	if filename.endswith('.csv'):
		filenames.append(filename)

for name in filenames:
	path = os.path.join("breath_data_optimal", name)
	data, target = create_data(pd.read_csv(filepath_or_buffer=path), name)
	data_list.append(data)
	target_list.append(target)

for i in range(17):
	plt.figure(i+1)
	plt.plot(data_list[i])
	#plt.figure(2)
	plt.plot(target_list[i])
	plt.show()
"""

