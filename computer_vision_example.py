# Simple example of a computer vision problem, using mnist data set

import tensorflow as tf
import matplotlib.pyplot as plt

# delcare call back function to minimize number of epochs required for training
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):  # on_epoch_end is called when the epoch ends
        if logs.get('loss') < 0.3:
            print("\nReached 70% accuracy so cancelling training!")
            self.model.stop_training = True


callbacks = myCallback()

# The Fashion MNIST data is available directly via the keras api
mnist = tf.keras.datasets.fashion_mnist

# Load training and testing data from mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# Plot the image
# plt.imshow(training_images[0])
# plt.show()

# normalize data
# non normalized data may have large scales (ranges) which skew the learning attempts
# 255 is the max value an image can have (test_images.max())
training_images = training_images / 255.0
test_images = test_images / 255.0

# Declare neural network
# Each additional layer requires an activation function to dictate their work
# Activation functions add the nonlinear component to the neural network layer
# without them, we would just be performing linear regression

# Flatten - reduce data dimension: 2d array (square) to 1d
# Relu - f(x) = x | x > 0; 0 | x <= 0
# Softmax - f(x) = f(x) | f(x) = max; 0 otherwise
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),  # hidden layer 1
                                    tf.keras.layers.Dense(256, activation=tf.nn.relu),  # hidden layer 2
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
# the final (output) layer should have as many neurons as distinct labels in the data set
# Additional hidden layers can be added to improve the modelling - particularly for complex problems
# one hidden layer is enough for this particular problem.
# note, model over fitting can occur with too many neurons/epochs

# declare training
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# cross entropy: the cross entropy between two probability distributions p and q over
# the same underlying set of events measures the average number of bits needed
# to identify an event drawn from the set if a coding scheme used for the set is optimized
# for an estimated probability distribution q, rather than the true distribution p
# P = training images (ground truth), q = model result


# train model
model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks])

# test model
model.evaluate(test_images, test_labels)

# predict image classification using model
classifications = model.predict(test_images)

# this shows the success probabilities of the first 10 predictions the model made
print('\n', classifications[0])
# there are 10 clothing items in the data set. Since the largest value in these predictions is the 10th element
# the corresponding item label (item 9) is the most probable
