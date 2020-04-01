import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Dense, LSTM, Dropout

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
###################################################################################################
### Function definitions
###
###################################################################################################

# Function to split global node dataframe into specific node dataframes
def dfNodeSplit(node):
	# Split dataframes into each node
	mask = global_data['Clave del nodo'] == node
	return global_data[mask]

# Returns prepared train and test. Dateparam indicates split between test and training data
def genTrainTestData(df,date_param):
	# Prepare data sets
	data_training = df[df['Fecha'] < date_param].copy()
	data_test = df[df['Fecha'] >= date_param].copy()
	
	# Clean dataframe
	training_data = data_training.drop(['Fecha','Clave del nodo','Hora'], axis=1)

	# Scale data 0-1
	scaler = MinMaxScaler()
	training_data = scaler.fit_transform(training_data)
	
	# Create training lists
	x_train = []
	y_train = []

	# Separate data DATA_steps
	for i in range(DATA_STEP,training_data.shape[0]):
		x_train.append(training_data[i-DATA_STEP:i])
		y_train.append(training_data[i,0])

	# Convert data into array
	x_train, y_train = np.array(x_train), np.array(y_train)

	return x_train, y_train, data_test, scaler, data_training

# Building LSTM NN
def genNN(x_train):

	regressor = Sequential()

	#Change units to improve model, as well as dropout
	#Layer 1
	regressor.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],4)))
	regressor.add(Dropout(0.1))

	#Layer 2
	regressor.add(LSTM(units=120, return_sequences=True))
	regressor.add(Dropout(0.1))

	#Layer 3
	regressor.add(LSTM(units=120, return_sequences=True))
	regressor.add(Dropout(0.1))

	#Layer 4
	regressor.add(LSTM(units=240))
	regressor.add(Dropout(0.1))

	#Layer 5
	regressor.add(Dense(units=1))

	return regressor

# Train NN: NN object, y training data, optimizer algorithm, loss algorithm, number of epochs, batch size
def trainNN(regressor, x_train, y_train, opt, l, ep, bat_s):
	#Train the model, may increase the epochs
	regressor.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
	regressor.fit(x_train, y_train, epochs=ep, batch_size=bat_s)
	return regressor

# Form test data for predicting
def testDataNN(DS, data_test,data_training,scaler):
	#Prepare test data set
	past_DATA_STEP_days = data_training.tail(DATA_STEP)
	df = past_DATA_STEP_days.append(data_test, ignore_index=True)
	df = df.drop(['Fecha','Clave del nodo','Hora'], axis=1)

	#scale data
	inputs = scaler.transform(df)

	#create test lists
	x_test = []
	y_test = []

	#Separate data DATA_steps
	for i in range(DATA_STEP, inputs.shape[0]):
		x_test.append(inputs[i-DATA_STEP:i])
		y_test.append(inputs[i,0])

	#convert data into array
	x_test, y_test = np.array(x_test), np.array(y_test)
	return x_test, y_test

# Predict prices based on the LSTM Model with the prepared data
def forecastLSTM(regressor, x_test, y_test, scaler, data_training):
	y_pred = regressor.predict(x_test)

	#Inverse scaling in the prediction
	scale = 1 / scaler.scale_[0]
	y_pred = y_pred * scale
	y_test = y_test * scale

	return y_pred, y_test

#Visualize data in graph form
def visualizeData(y_test, y_pred, node):
	plt.figure(figsize=(14,5))
	plt.plot(y_test, color='red', label='Real LMP')
	plt.plot(y_pred, color='blue', label='Predicted LMP')
	plt.title(node)
	plt.xlabel('Time')
	plt.ylabel('LMP')
	plt.legend()
	plt.show()

###################################################################################################
### Main Program
###
###################################################################################################

### Parameter definition
frames = []
DATA_STEP = 128
EPOCHS = 10
BATCH_SIZE = 64

### Read files and form global dataframe
# Iterate through all files and add node data to global dataframe
for filename in os.listdir('PML_DATA_selected_nodes/'):
	print("Gathering info from: " + filename)
	URI = 'PML_DATA_selected_nodes/'+filename
	temp = pd.read_csv(URI, date_parser=True)
	frames.append(temp)
global_data = pd.concat(frames, sort=False)



### Main program, call functions to perform LSTM prediction.

# High congestion node
hc_node = '08CHS-34.5'
hc_df = dfNodeSplit(hc_node)
hc_x_train, hc_y_train, hc_data_test, hc_scaler, hc_data_training = genTrainTestData(hc_df,'2019-12-31')
hc_regressor = genNN(hc_x_train)
hc_regressor = trainNN(hc_regressor,hc_x_train, hc_y_train, 'adam','mean_squared_error',EPOCHS, BATCH_SIZE)
hc_x_test, hc_y_test = testDataNN(DATA_STEP, hc_data_test,hc_data_training, hc_scaler)
hc_y_pred, hc_y_test = forecastLSTM(hc_regressor, hc_x_test, hc_y_test, hc_scaler, hc_data_training)
visualizeData(hc_y_test,hc_y_pred, hc_node)

# Medium congestion node
mc_node = '03STG-115'
mc_df = dfNodeSplit(mc_node)
mc_x_train, mc_y_train, mc_data_test, mc_scaler, mc_data_training = genTrainTestData(mc_df,'2019-12-31')
mc_regressor = genNN(mc_x_train)
mc_regressor = trainNN(mc_regressor, mc_x_train, mc_y_train, 'adam','mean_squared_error',EPOCHS, BATCH_SIZE)
mc_x_test, mc_y_test = testDataNN(DATA_STEP, mc_data_test,mc_data_training, mc_scaler)
mc_y_pred, mc_y_test = forecastLSTM(mc_regressor,mc_x_test, mc_y_test, mc_scaler, mc_data_training)
visualizeData(mc_y_test,mc_y_pred, mc_node)

# Low Congestion node
lc_node = '04PLD-230'
lc_df = dfNodeSplit(lc_node)
lc_x_train, lc_y_train, lc_data_test, lc_scaler, lc_data_training = genTrainTestData(lc_df,'2019-12-31')
lc_regressor = genNN(lc_x_train)
lc_regressor = trainNN(lc_regressor, lc_x_train, lc_y_train, 'adam','mean_squared_error',EPOCHS, BATCH_SIZE)
lc_x_test, lc_y_test = testDataNN(DATA_STEP, lc_data_test, lc_data_training, lc_scaler)
lc_y_pred, lc_y_test = forecastLSTM(lc_regressor,lc_x_test, lc_y_test, lc_scaler, lc_data_training)
visualizeData(lc_y_test,lc_y_pred, lc_node)







