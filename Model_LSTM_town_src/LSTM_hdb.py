''' Adpoted from machinelearningmastery.com '''

import pandas as pd
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
import numpy
import pickle
 

 
# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
	df = DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df
 
# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)
 
# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]
 
# scale train and test data to [-1, 1]
def scale(train, test = None):
	# fit scaler
	scaler = MinMaxScaler()
	scaler.partial_fit(train)
	if test is not None:
		scaler.partial_fit(test)
	# transform train
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	# transform test
	if test is None:
		return scaler, train_scaled
	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled
 
# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value]
	array = numpy.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]
 
# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	model = Sequential()
	model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='RMSprop')
	model.fit(X, y, epochs=nb_epoch, batch_size=batch_size, verbose=2, shuffle=False)
	return model
 
# make a one-step forecast
def forecast_lstm(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]


val = False

# load dataset
series = read_csv('df_hdb_mean_time.csv', header=0, index_col=0, squeeze=True).set_index('monthtx').squeeze()


#print([town for town in Towns for i in range(3)])

out = pd.DataFrame({'monthtx': [24214,24215]})

predictions = list()


# transform data to be stationary
raw_values = series.values
diff_values = difference(raw_values, 1)
 
# transform data to be supervised learning
supervised = timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values


# split data into train and test-sets
if val:
	num_tested = 7
	train, test = supervised_values[0:-1*num_tested], supervised_values[-1*num_tested:]
else: 
	look_ahead = 2
	train = supervised_values
 
# transform the scale of the data
if val:
	scaler, train_scaled, test_scaled = scale(train, test)
else:
	scaler, train_scaled = scale(train)
 
# fit the model
lstm_model = fit_lstm(train_scaled, 1, 10, 4)
# forecast the entire training dataset to build up state for forecasting
train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
lstm_model.predict(train_reshaped, batch_size=1)
 
# walk-forward validation on the test data
if val:
	for i in range(len(test_scaled)):
		# make one-step forecast
		X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
		yhat = forecast_lstm(lstm_model, 1, X)
		# invert scaling
		yhat = invert_scale(scaler, X, yhat)
		# invert differencing
		yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
		# store forecast
		predictions.append(yhat)
		expected = raw_values[len(train) + i + 1]
		print('Month=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))
else:
	inputs = [train_scaled[-1,1:2]]
	for i in range(look_ahead):
		yhat = forecast_lstm(lstm_model, 1, inputs[i])
		inputs.append(numpy.array([yhat]))
		yhat = invert_scale(scaler, inputs[i], yhat)
		yhat = inverse_difference(raw_values, yhat, look_ahead+1-i)
		predictions.append(yhat)
		print('Month=%d, Predicted=%f' % (i+1, yhat))
 
# report performance
if val:
	mae = mean_absolute_error(raw_values[-1*num_tested:], predictions)
	print('Test MAE: %.3f' % mae)
	# line plot of observed vs predicted
	pyplot.plot(raw_values[-num_tested:])
	pyplot.plot(predictions)
else:
	pyplot.plot(numpy.append(raw_values,predictions))
#pyplot.show()

if not val:
	out['mean_resale_price'] = predictions
	out.to_csv('pred_mean_hdb.csv')

