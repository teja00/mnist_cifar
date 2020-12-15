def normalize(X):
    return X/255
from keras.datasets import mnist
import keras
import keras.utils
#from keras import utils as np_utils
import matplotlib.pyplot as plt
(trainX,trainY),(testX,testY) = mnist.load_data()
trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
testX = testX.reshape((testX.shape[0], 28, 28, 1))

trainY = keras.utils.to_categorical(trainY)
testY = keras.utils.to_categorical(testY)
trainX = normalize(trainX)
testX = normalize(testX)

from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Flatten,Dense, Dropout, Activation
model = keras.Sequential()
model.add(Convolution2D(32, (7,7), activation = 'relu', input_shape = (28,28,1)))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Convolution2D(64, (9,9), activation = 'relu', input_shape = (28,28,1)))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))
OPT = keras.optimizers.SGD(learning_rate = 0.01, momentum = 0.9)
Epochs_train = []
Epochs_Val = []

model.compile(loss='categorical_crossentropy',optimizer= 'adadelta', metrics=['accuracy'])

epoch = model.fit(trainX, trainY, epochs = 100, validation_data = (testX,testY))
model.evaluate(testX,testY)


plt.plot(epoch.history['val_accuracy'])
plt.plot(epoch.history['accuracy'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train','Test'])


import pandas as pd
Accuracies = pd.DataFrame()
Accuracies['Validation'] = epoch.history['val_accuracy']
Accuracies['Training'] = epoch.history['accuracy']
writer = pd.ExcelWriter('Accuracies.xlsx')
Accuracies.to_excel(writer)


from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)