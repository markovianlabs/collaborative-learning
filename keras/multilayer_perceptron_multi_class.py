from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np 


# intialize the model.
model = Sequential()

## Dense(64) is fully-connected layer with 64 hidden units.
## In the first layer, we must specify the expected number of input dat shape, e.g. 20d vectors.
model.add(Dense(64, input_dim=20, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))

model.add(Dense(64, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))

model.add(Dense(10, init='uniform'))
model.add(Activation('softmax'))


## optimizer
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

## compilation
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

## data
X_train = np.random.random((1000, 20))
X_test = np.random.random((100, 20))
y_train = np.random.randint(5, size=(1000, 10))
y_test = np.random.randint(5, size=(100, 10))


## Training
model.fit(X_train, y_train, nb_epoch=20, batch_size=16)

## Evaluation
score = model.evaluate(X_test, y_test, batch_size=16)
print score


