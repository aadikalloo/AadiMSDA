import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import numpy as np
from keras.optimizers import SGD

nb_classes = 10

# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
img_channels = 1
img_rows = 28
img_cols = 28

X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

num_conv_filters_layer1 = 16
num_conv_filters_layer2 = 16
num_conv_kernel_rows, num_conv_kernel_cols = 3, 3

model = Sequential()
act = 'relu' #relu
model.add(Convolution2D(num_conv_filters_layer1, num_conv_kernel_rows, num_conv_kernel_cols, border_mode='same', input_shape=(img_channels, img_rows, img_cols)))
model.add(Activation(act))
model.add(Convolution2D(num_conv_filters_layer1, num_conv_kernel_rows, num_conv_kernel_cols))
model.add(Activation(act))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.7)) #0.25

model.add(Convolution2D(num_conv_filters_layer2, num_conv_kernel_rows, num_conv_kernel_cols, border_mode='same'))
model.add(Activation(act))
model.add(Convolution2D(num_conv_filters_layer2, num_conv_kernel_rows, num_conv_kernel_cols))
model.add(Activation(act))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.7)) #0.25

model.add(Flatten())
model.add(Dense(512)) #model.add(Dense(512))
model.add(Dropout(0.8)) #0.5
model.add(Activation(act))
model.add(Dense(128)) #model.add(Dense(512)) #added
model.add(Dropout(0.5)) #0.5 #added
model.add(Activation(act)) #added
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='sgd')

batch_size = 8
nb_epoch = 20
learning_rate = 0.001
sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
