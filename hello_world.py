# Simple "hello world"-esque to demonstrate neural networking using keras

import tensorflow as tf
import numpy as np
from tensorflow import keras

# declare network
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# declare learning parameters
# sgd = stochastic gradient descent (stochastic because gradient is performed on random subset of builder functions)
model.compile(optimizer='sgd', loss='mean_squared_error')

#  declare training data.
xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

print('Training the model...')
# train the neural network. Epochs = # reps. Set verbose=0 to remove fit output
model.fit(xs, ys, epochs=250, verbose=0)

# test how well the model is trained
print('\n')
print('Testing model fit for function: f(x) = 2x+1\n')
print('f(10) = {:.3f}'.format(model.predict([10.0])[0][0]))
