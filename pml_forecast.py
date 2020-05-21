import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.utils import check_array
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Dense, LSTM, Dropout
from math import sqrt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
###################################################################################################
### Function definitions
###
###################################################################################################

# Function to split global node dataframe into specific node dataframes
def dfNodeSplit(node):
	# Split dataframes into each node
	mask = global_data['Clave del nodo'] == node
	return_df = global_data[mask]
	return_df.set_index(['Fecha','Hora'], inplace=True,drop=False)


	return return_df

# Returns prepared train and test. Dateparam indicates split between test and training data
def genTrainTestData(df,date_param):
	# Prepare data sets
	data_training = df[df['Fecha'] < date_param].copy()
	data_test = df[df['Fecha'] >= date_param].copy()
	data_test = data_test.drop(['Fecha','Clave del nodo','Hora'], axis='columns')
	
	# dates = data_test.copy()
	# dates = dates.drop(['Clave del nodo','Precio marginal local ($/MWh)','Componente de energia ($/MWh)', 'Componente de perdidas ($/MWh)', 'Componente de congestion ($/MWh)'], axis=1)
	# index = dates

	# Clean dataframe
	training_data = data_training.drop(['Fecha','Clave del nodo','Hora'], axis=1)
	#training_data = data_training.drop(['Clave del nodo', 'Componente de energia ($/MWh)', 'Componente de perdidas ($/MWh)', 'Componente de congestion ($/MWh)'], axis=1)
	
	print(training_data.head())

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
	regressor.add(LSTM(units=40, return_sequences=True, input_shape=(x_train.shape[1],4)))
	

	#Layer 2
	regressor.add(LSTM(units=120, return_sequences=True))
	#regressor.add(Dropout(0.005))

	# #Layer 3
	# regressor.add(LSTM(units=120, return_sequences=True))
	# regressor.add(Dropout(0.1))

	#Layer 4
	regressor.add(LSTM(units=240))
	regressor.add(Dropout(0.05))

	#Layer 5
	regressor.add(Dense(units=1))

	return regressor

	

# Train NN: NN object, y training data, optimizer algorithm, loss algorithm, number of epochs, batch size
def trainNN(regressor, x_train, y_train, opt, l, ep, bat_s):
	#Train the model, may increase the epochs
	regressor.compile(optimizer='adam', loss='mean_squared_error')
	history = regressor.fit(x_train, y_train, epochs=ep, batch_size=bat_s, validation_split=0.33)

	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model train vs validation loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'validation'], loc='upper right')
	plt.show()

	return regressor

# Form test data for predicting
def testDataNN(DS, data_test,data_training,scaler):
	#Prepare test data set
	past_DATA_STEP_days = data_training.tail(DATA_STEP)
	#df = past_DATA_STEP_days.append(data_test, ignore_index=True)

	df = past_DATA_STEP_days.append(data_test,sort=False)

	
	df = df.drop(['Fecha','Clave del nodo','Hora'], axis=1)
	#df = data_test.drop(['Clave del nodo','Componente de energia ($/MWh)', 'Componente de perdidas ($/MWh)', 'Componente de congestion ($/MWh)'], axis=1)

	#df.set_index(['Fecha','Hora'], inplace=True)

	#scale data
	inputs = scaler.transform(df)
	#inputs = scaler.transform(data_test)

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

# Gets historical average for prices and returns a tuple with the hour in which electricity is cheaper (based on avg)
# and its price.
def historicalAverage(node):
	# Create return dictionary
	avgPML = {}
	# Clean dataframe
	clean_df = node.drop(['Fecha','Clave del nodo'], axis=1)

	#Split df into hours
	for i in range(1,25):
		mask = clean_df['Hora'] == i
		avgPML[i] = clean_df[mask]
	
	#Get average for each hour
	for i in range(1,25):
		avgPML[i] = avgPML[i].mean(0)
	
	key_min = min(avgPML.keys(), key=(lambda k: avgPML[k]['Precio marginal local ($/MWh)']))
	return (key_min, avgPML[key_min]['Precio marginal local ($/MWh)'])
	


# Buying routine: LSTM will look forward 24 hours to find minimum price of the day, historical avg
# will buy the default hr.
#for date in data.index.get_level_values('Fecha'):
def simulation(data,avg_hr):
	

	print('Min Precio marginal local ($/MWh)')
	print(len(data.loc[(slice(None),5), :]))
	print(data.loc[(slice(None),5), :].sum()['Precio marginal local ($/MWh)'])
	#print(data.groupby(level='Fecha').idxmin()['Precio marginal local ($/MWh)'])


	print('SUM Min Prediccion LSTM PML')
	lstm_min_index = data.groupby(level='Fecha').idxmin()['Prediccion PML'].tolist()
	sum_lstm_PML = 0
	for i in lstm_min_index:
		#print(data.loc[i]['Precio marginal local ($/MWh)'],data.loc[i]['Prediccion PML'])
		sum_lstm_PML += data.loc[i]['Precio marginal local ($/MWh)']
	print(sum_lstm_PML)
	#print(data.iloc[data.index.get_level_values('Fecha') == '2020-01-01'])

	print("Absolute minimum")
	print(data.groupby(level='Fecha').min()['Precio marginal local ($/MWh)'].sum())


def mean_absolute_percentage_error(y_true, y_pred): 
    #y_true, y_pred = check_array(y_true, y_pred)

    ## Note: does not handle mix 1d representation
    #if _is_1d(y_true): 
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

#Print statistic metrics for each prediction.
def statistics(real_Data,predictions):
	scaler = MinMaxScaler()
	print(real_Data)
	print(predictions)
	print("Root Mean squared error")
	print(sqrt(mean_squared_error(real_Data.reshape(-1, 1),predictions)))
	print("Mean Absolute Error")
	print(mean_absolute_error(real_Data.reshape(-1,1), predictions))
	print("Mean Absolute Percentage Error")
	print(mean_absolute_percentage_error(real_Data.reshape(-1,1), predictions))


###################################################################################################
### Main Program
###
###################################################################################################

### Parameter definition
frames = []
DATA_STEP = 72
EPOCHS = 10
BATCH_SIZE = 168

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
hc_node = '08COZ-34.5'
hc_df = dfNodeSplit(hc_node)
hc_x_train, hc_y_train, hc_data_test, hc_scaler, hc_data_training = genTrainTestData(hc_df,'2019-12-31')
print("debug")
print(len(hc_x_train))
print(hc_x_train)
hc_regressor = genNN(hc_x_train)
hc_regressor = trainNN(hc_regressor,hc_x_train, hc_y_train, 'adam','mean_squared_error',EPOCHS, BATCH_SIZE)
hc_x_test, hc_y_test = testDataNN(DATA_STEP, hc_data_test,hc_data_training, hc_scaler)
hc_y_pred, hc_y_test = forecastLSTM(hc_regressor, hc_x_test, hc_y_test, hc_scaler, hc_data_training)

# Add prediction to test dataframe
hc_data_test['Prediccion PML'] = hc_y_pred
#print(hc_data_test)

hc_cheapest_hr = historicalAverage(hc_data_training)

#Simulation: Historical average against LSTM
simulation(hc_data_test,hc_cheapest_hr)
visualizeData(hc_y_test,hc_y_pred, hc_node)
statistics(hc_y_test,hc_y_pred)

# Medium congestion node
mc_node = '04EFU-115'
mc_df = dfNodeSplit(mc_node)
mc_x_train, mc_y_train, mc_data_test, mc_scaler, mc_data_training = genTrainTestData(mc_df,'2019-12-31')
mc_regressor = genNN(mc_x_train)
mc_regressor = trainNN(mc_regressor, mc_x_train, mc_y_train, 'adam','mean_squared_error',EPOCHS, BATCH_SIZE)
mc_x_test, mc_y_test = testDataNN(DATA_STEP, mc_data_test,mc_data_training, mc_scaler)
mc_y_pred, mc_y_test = forecastLSTM(mc_regressor,mc_x_test, mc_y_test, mc_scaler, mc_data_training)

# Add prediction to test dataframe
mc_data_test['Prediccion PML'] = mc_y_pred
#print(hc_data_test)

mc_cheapest_hr = historicalAverage(mc_data_training)

#Simulation: Historical average against LSTM
simulation(mc_data_test,mc_cheapest_hr)
visualizeData(mc_y_test,mc_y_pred, mc_node)
statistics(mc_y_test,mc_y_pred)

# Low Congestion node
lc_node = '06PUO-115'
lc_df = dfNodeSplit(lc_node)
lc_x_train, lc_y_train, lc_data_test, lc_scaler, lc_data_training = genTrainTestData(lc_df,'2019-12-31')
lc_regressor = genNN(lc_x_train)
lc_regressor = trainNN(lc_regressor, lc_x_train, lc_y_train, 'adam','mean_squared_error',EPOCHS, BATCH_SIZE)
lc_x_test, lc_y_test = testDataNN(DATA_STEP, lc_data_test, lc_data_training, lc_scaler)
lc_y_pred, lc_y_test = forecastLSTM(lc_regressor,lc_x_test, lc_y_test, lc_scaler, lc_data_training)


# Add prediction to test dataframe
lc_data_test['Prediccion PML'] = lc_y_pred
#print(hc_data_test)

lc_cheapest_hr = historicalAverage(lc_data_training)

#Simulation: Historical average against LSTM
simulation(lc_data_test,lc_cheapest_hr)
visualizeData(lc_y_test,lc_y_pred, lc_node)
statistics(lc_y_test,lc_y_pred)