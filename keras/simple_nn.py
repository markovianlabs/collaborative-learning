# Start by importing numpy as Keras, like scikit-learn, requires data to be a numpy array.
import numpy as np 

# We will use the MNIST dataset containing hand-written digits.
from sklearn import datasets
from sklearn.cross_validation import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils

# Set seed for reproducibility.
np.random.seed(1234)

# Load data
mnist = datasets.load_digits()
X = mnist.data
Y = mnist.target

# let's examine data
print X.shape, Y.shape
print X[0]
print Y[0]

# Split the data into a training and test set.
train_X, test_X, train_y, test_y = train_test_split(X, Y, train_size=0.7, random_state=0)

# One Hot Encoding
def one_hot_encode_object_array(arr):
    '''One hot encode a numpy array of objects (e.g. strings)'''
    uniques, ids = np.unique(arr, return_inverse=True)
    return np_utils.to_categorical(ids, len(uniques))

# One hot encode labels for training and test sets.
train_y_ohe = one_hot_encode_object_array(train_y)
test_y_ohe = one_hot_encode_object_array(test_y)


# Initialize a Sequential neural network.
model = Sequential()

# Add a fully connected layer with 32 units. Each unit recieves an input from every unit in the input layer, and since number of units in the input is equal to the dimension (64) of the input vectors, we need the input shape to be 64.
# Keras uses Dense to create a fully connected layer.
model.add(Dense(32, input_shape=(64,)))

# Add an activation after the first layer. We will use sigmoid activation, which is DEFN. Other choices like relu etc are also possible.
model.add(Activation('sigmoid'))

# Add another layer
# model.add(Dense(16))
# model.add(Activation('relu'))

# Add output layer, which is always fully connected. Since the output will be OHE labels, which are 10 dimensions, we would want the output layer to have 10 units.
model.add(Dense(10))


# Add activation for the output layer. In classification tasks, we use softmax activation, which is -- DEFN--. This provides a probilistic interpretation for the output labels...more details..
model.add(Activation('softmax'))

# Next we need to configure the model. There are some more choices we need to make before we can run the model.
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
# optimizer
# loss
# metrics

# Train the network. fit method, similar to sklearn.
model.fit(train_X, train_y_ohe, nb_epoch=10, batch_size=30)
# nb_epoch, batch_size

# Compute accuracy of the model based on this network.
# print model.predict(test_X)[0]
loss, accuracy = model.evaluate(test_X, test_y_ohe)
print accuracy

