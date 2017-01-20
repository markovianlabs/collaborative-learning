from __future__ import print_function

import numpy as np 

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K


# Set the seed
np.random.seed(1234)

# Set params
batch_size = 128
nb_classes = 10
nb_epoch = 12

# image dimensions
img_rows, img_cols = 28, 28

# conv filters
nb_filters = 32

# size of pooling area for max pooling
pool_size = (2,2)

# conv kernal size
kernal_size = (3,3)

# data
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

print(X_train.shape)
print(X_test.shape)
print(X_train[0].shape)
print(Y_train.shape)
# print(K)

# image odering convention - theano or tensorflow
if (K.image_dim_ordering() == 'th'):
	X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
	X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
	input_shape = (1, img_rows, img_cols)
else:
	X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
	X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
	input_shape = (img_rows, img_cols, 1)


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

print('X_train shape: ', X_train.shape)
print('X_test shape: ', X_test.shape)



# convert class vectors to binary class matrices (OHE)
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)

# Model
model = Sequential()

model.add(Convolution2D(nb_filters, kernal_size[0], kernal_size[1], border_mode='valid', input_shape=input_shape))
model.add(Activation('relu'))

model.add(Convolution2D(nb_filters, kernal_size[0], kernal_size[1]))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=["accuracy"])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose=0)

print('Test Score: ', score[0])
print('Test Accuracy: ', score[1])









