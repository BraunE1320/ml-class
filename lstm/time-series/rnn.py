import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import LSTM, SimpleRNN, Dropout
from keras.callbacks import LambdaCallback

import wandb
from wandb.keras import WandbCallback

import plotutil
from plotutil import PlotCallback

wandb.init()
config = wandb.config

<<<<<<< ba7f2d2e6a7c4e75ed7b805c82daf4f476afaa0e
<<<<<<< dd0d09abdb3fc4c5d5512738c60354db5fe3ad82
config.repeated_predictions = False
config.look_back = 25
=======
config.repeated_predictions = True
config.look_back = 4
>>>>>>> translation in lstm
=======
<<<<<<< 90da4f70b2afd5f40e0597a02c23d04078d5e12b
config.repeated_predictions = True
config.look_back = 4
=======
config.repeated_predictions = False
config.look_back = 25
>>>>>>> Updating basic networks
>>>>>>> Updating basic networks

def load_data(data_type="airline"):
    if data_type == "flu":
        df = pd.read_csv('flusearches.csv')
        data = df.flu.astype('float32').values
    elif data_type == "airline":
        df = pd.read_csv('international-airline-passengers.csv')
        data = df.passengers.astype('float32').values
    elif data_type == "sin":
        df = pd.read_csv('sin.csv')
        data = df.sin.astype('float32').values
    return data

# convert an array of values into a dataset matrix
def create_dataset(dataset):
    dataX, dataY = [], []
    for i in range(len(dataset)-config.look_back-1):
        a = dataset[i:(i+config.look_back)]
        dataX.append(a)
        dataY.append(dataset[i + config.look_back])
    return np.array(dataX), np.array(dataY)

data = load_data("sin")
    
# normalize data to between 0 and 1
max_val = max(data)
min_val = min(data)
data=(data-min_val)/(max_val-min_val)

# split into train and test sets
split = int(len(data) * 0.70)
train = data[:split]
test = data[split:]

trainX, trainY = create_dataset(train)
testX, testY = create_dataset(test)

trainX = trainX[:, :, np.newaxis]
testX = testX[:, :, np.newaxis]

# create and fit the RNN
model = Sequential()
<<<<<<< ba7f2d2e6a7c4e75ed7b805c82daf4f476afaa0e
<<<<<<< dd0d09abdb3fc4c5d5512738c60354db5fe3ad82
model.add(SimpleRNN(7, input_shape=(config.look_back,1 )))
model.add(Dense(7, activation="tanh"))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='mse', optimizer='adam')
model.fit(trainX, trainY, epochs=500, batch_size=30, validation_data=(testX, testY),  callbacks=[WandbCallback(), PlotCallback(trainX, trainY, testX, testY, config.look_back)])
=======
=======
<<<<<<< 90da4f70b2afd5f40e0597a02c23d04078d5e12b
>>>>>>> Updating basic networks
model.add(SimpleRNN(1, input_shape=(config.look_back,1 )))
model.compile(loss='mae', optimizer='rmsprop')
model.fit(trainX, trainY, epochs=1000, batch_size=20, validation_data=(testX, testY),  callbacks=[WandbCallback(), PlotCallback(trainX, trainY, testX, testY, config.look_back)])




<<<<<<< ba7f2d2e6a7c4e75ed7b805c82daf4f476afaa0e
>>>>>>> translation in lstm
=======
=======
model.add(SimpleRNN(7, input_shape=(config.look_back,1 )))
model.add(Dense(7, activation="tanh"))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='mse', optimizer='adam')
model.fit(trainX, trainY, epochs=500, batch_size=30, validation_data=(testX, testY),  callbacks=[WandbCallback(), PlotCallback(trainX, trainY, testX, testY, config.look_back)])
>>>>>>> Updating basic networks
>>>>>>> Updating basic networks

