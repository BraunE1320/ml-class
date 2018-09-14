# A very simple perceptron for classifying american sign language letters
import signdata
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout,BatchNormalization,Conv2D,MaxPooling2D,AveragePooling2D,ZeroPadding2D
from keras.utils import np_utils
import wandb
from wandb.keras import WandbCallback

# logging code
run = wandb.init()
config = run.config
config.loss = "categorical_crossentropy"
config.optimizer = "adam"
config.epochs = 20
config.first_layer_conv_height = 3
config.first_layer_conv_width = 3

# load data
(X_test, y_test) = signdata.load_test_data()
(X_train, y_train) = signdata.load_train_data()

img_width = X_test.shape[1]
img_height = X_test.shape[2]

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_train.shape[1]

# you may want to normalize the data here..
X_test = X_test.astype('float32') / 255.
X_train = X_train.astype('float32') / 255.

#reshape input data
X_train = X_train.reshape(X_train.shape[0], img_width, img_height, 1)
X_test = X_test.reshape(X_test.shape[0], img_width, img_height, 1)

# create model
model=Sequential()
model.add(Conv2D(128,
    (config.first_layer_conv_width, config.first_layer_conv_height),
    input_shape=(img_width, img_height,1),
    activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128,
    (config.first_layer_conv_width, config.first_layer_conv_height),
    activation='relu'))
model.add(Conv2D(128,
    (config.first_layer_conv_width, config.first_layer_conv_height),
    activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64,
    (config.first_layer_conv_width, config.first_layer_conv_height),
    activation='relu'))
model.add(Conv2D(64,
    (config.first_layer_conv_width, config.first_layer_conv_height),
    activation='relu'))
model.add(Dropout(0.4))
model.add(Conv2D(64,
    (config.first_layer_conv_width, config.first_layer_conv_height),
    activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Conv2D(32,
    (config.first_layer_conv_width, config.first_layer_conv_height),
    activation='relu'))
model.add(Conv2D(32,
    (config.first_layer_conv_width, config.first_layer_conv_height),
    activation='relu'))
model.add(Dropout(0.4))
model.add(Conv2D(32,
    (config.first_layer_conv_width, config.first_layer_conv_height),
    activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=config.loss, optimizer=config.optimizer,
                metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, epochs=config.epochs, validation_data=(X_test, y_test),
                    callbacks=[WandbCallback(data_type="image", labels=signdata.letters)])
