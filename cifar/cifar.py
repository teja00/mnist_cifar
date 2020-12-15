from keras.datasets import cifar10
import keras
import keras.utils
(trainX, trainY), (testX, testY) = cifar10.load_data()
trainY = keras.utils.to_categorical(trainY)
testY = keras.utils.to_categorical(testY)
def normalize(X):
    return X/255
trainX = normalize(trainX)
testX = normalize(testX)

from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Flatten, Activation, Dropout

model = keras.Sequential()
model.add(Convolution2D(32, (3,3), activation = 'relu', input_shape = (32,32,3) ) )
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Convolution2D(64,(3,3),activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))
model.add(Convolution2D(64,(3,3),activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation ('softmax'))

OPT = keras.optimizers.SGD(learning_rate = 0.001, momentum = 0.9)
model.compile(loss = 'categorical_crossentropy', optimizer = OPT, metrics = ['accuracy'])

epoch = model.fit(trainX, trainY, epochs = 250, validation_data = (testX,testY))
model.evaluate(testX,testY)

from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

import matplotlib.pyplot as plt

plt.plot(epoch.history['val_accuracy'])
plt.plot(epoch.history['accuracy'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Test','Train'])

import pandas as pd
Accuracies = pd.DataFrame()
Accuracies['Validation'] = epoch.history['val_accuracy']
Accuracies['Training'] = epoch.history['accuracy']
Accuracies.to_excel("accuracy_cifar.xlsx")


from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)