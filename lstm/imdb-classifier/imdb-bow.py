import imdb
import numpy as np
from keras.models import Sequential
from keras.preprocessing import text
<<<<<<< 90da4f70b2afd5f40e0597a02c23d04078d5e12b
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.layers import Flatten, Dense, Dropout
import wandb
from wandb.keras import WandbCallback
from sklearn.linear_model import LogisticRegression

wandb.init()
config = wandb.config
config.vocab_size = 1000

# X_train = txt review files
# y_train = 0 for neg, 1 for pos
(X_train, y_train), (X_test, y_test) = imdb.load_imdb()

tokenizer = text.Tokenizer(num_words=config.vocab_size)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_matrix(X_train)
X_test = tokenizer.texts_to_matrix(X_test)

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# create model
model=Sequential()
model.add(Dense(2, activation="softmax", input_shape=(1000,)))
model.compile(loss='binary_crossentropy', optimizer='adam',
                metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test),
                    callbacks=[WandbCallback(save_model=False)])
#img_width = X_train.shape[0]
#img_height = X_train.shape[1]
#print(X_train.shape)
#exit()

model = Sequential()
model.add(Dense(1, activation="relu", input_shape=(1000,)))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test),
        epochs=10,
        callbacks=[WandbCallback(save_model=False)])

#bow_model = LogisticRegression()
#bow_model.fit(X_train, y_train)

#pred_train = bow_model.predict(X_train)
#acc = np.sum(pred_train==y_train)/len(pred_train)

#pred_test = bow_model.predict(X_test)
#val_acc = np.sum(pred_test==y_test)/len(pred_test)
#wandb.log({"val_acc": val_acc, "acc": acc})
