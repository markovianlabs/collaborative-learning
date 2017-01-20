## Sequential model
A *sequential model* is a linear stack of layers.

```
from keras.models import Sequential
from keras.layers import Dense, Activation

# Initialize the sequential model. Creates an instance of the constructor.
model =  Sequential()

# Add a fully-connected layer with 32 units. The dimensio of each input is 124.
model.add(Dense(32, input_dim=124))

# Add activation function, here RELU.
model.add(Activation('relu'))
```
- Dense
- Activation


- Input can be specified as a tuple using *input_shape* argument as 
```
model.add(Dense(32, input_shape=(124,)))
```

After we have a model set, we need to configure it before we can start training. This essentially means choosing an optimization method, a loss function and a metric for scoring. Keras does this using *compile* method.
```
model.compile(optimizer=chosen_optimizer, loss=chosen_loss, metrics=[chosen_metric])
```
e.g. for a binary classification problem
```
model.compile(optimizer='rmsprop', loss='binary_corssentropy, metrics=['accuracy'])
```

We are now ready to train the neural network. Keras has a *fit* method, similar to scikit-learn.

```
model.fit(data, labels, nb_epoch=10, batch_size=32)
```
- data: numpy array (vectors of features), labels: numpy array of labels.
- nb_epoch
- batch_size
- 
