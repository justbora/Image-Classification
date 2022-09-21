from traceback import print_tb
import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt

# load pre-defined dataset (70k of 28x28)
fashion_mnist=keras.datasets.fashion_mnist

# Pull out data from dataset
(train_images,train_labels),(test_images, test_labels)=fashion_mnist.load_data()

# Show data
# print(train_labels[0])
# print(train_images[0])
plt.imshow(train_images[1], cmap='gray',vmin=0,vmax=255)
plt.show()

# Define our nueral net structure:
model=keras.Sequential([
    # input is a 28x18 image ("flatten" flattens the 28x28 into a single 784x1 input)
    keras.layers.Flatten(input_shape=(28,28)),
     
    # Hidden layer is 128 deep.relu returns the value, or 0 (works good enough, much faster) 
    keras.layers.Dense(units=128,activation=tf.nn.relu),
     
    # output layer is 0-10 (depending om what piece of clothing it is). return maximum
    keras.layers.Dense(units=10,activation=tf.nn.softmax)
])

# compile our code 
model.compile(optimizer=tf.optimizers.Adam(),loss='sparse_categorical_crossentropy')

# train our model, using our training data set
model.fit(train_images, train_labels, epochs=5)

# test our model, using out test data set
test_loss=model.evaluate(test_images,test_labels)

plt.imshow(test_images[1],cmap='gray',vmin='0',vmax='255')
plt.show()

print(test_labels[1])

# Make pridictions
predictions=model.predict(test_images)

print(list(predictions[1]).index(max(predictions[1])))
if (test_labels[1]==list(predictions[1]).index(max(predictions[1]))):
    print('pridictions are correct')

print()
print(predictions[1])