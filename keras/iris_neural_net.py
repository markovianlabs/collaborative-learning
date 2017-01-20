import numpy as np 

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegressionCV
import sklearn.datasets as datasets

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils


iris = datasets.load_iris()
X = iris.data
Y = iris.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4)

lr = LogisticRegressionCV()
lr.fit(X_train, Y_train)
print lr.score(X_test, Y_test)


def one_hot_encode_object_array(arr):
	'''one hot encode a numpy array of objects e.g. strings'''
	uniques, ids = np.unique(arr, return_inverse=True)
	return np_utils.to_categorical(ids, len(uniques))


Y_train_ohe = one_hot_encode_object_array(Y_train)
Y_test_ohe = one_hot_encode_object_array(Y_test)

model = Sequential()
model.add(Dense(16, input_shape=(4,)))
model.add(Activation('sigmoid'))

model.add(Dense(3))
model.add(Activation('softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])

model.fit(X_train, Y_train_ohe, nb_epoch=100, batch_size=1, verbose=0)

loss, accuracy = model.evaluate(X_test, Y_test_ohe, verbose=0)
print accuracy