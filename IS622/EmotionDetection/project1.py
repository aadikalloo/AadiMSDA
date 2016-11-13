import pandas as pd
import numpy as np
import joblib
import multiprocessing
import cv2
import math
import random
num_cores = multiprocessing.cpu_count()

def readImages(index, file, image_path):
	file1 = image_path + file['image']
	img0 = cv2.imread(file1) #read image
	img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY) #convert image to grayscale
	return (img0, file['image'], file['emotion'])

def load_data(data_labels, img_loc):
	data_labels = data_labels.iloc[1:len(data_labels),:]
	image_array = joblib.Parallel(n_jobs=num_cores)(joblib.delayed(readImages)(index, image, img_loc) for index, image in data_labels.iterrows())
	image_array = np.asarray(image_array)

	y = [s for s in image_array if s[0].shape == (350,350)]
	y = np.asarray(y)
	fname_array = y[...,1]
	emotion_array = y[...,2]
	image_array = reshape_image_array(y)
	random.seed(1149)
	shuffled_indices = np.random.permutation(np.arange(len(fname_array)))

	shuffled_images = image_array[shuffled_indices]
	shuffled_fname_array = fname_array[shuffled_indices]
	shuffled_emotion_array = emotion_array[shuffled_indices]
	shuffled_data = pd.DataFrame(shuffled_fname_array, columns = ['Filename'])
	shuffled_data['Emotion'] = shuffled_emotion_array
	return shuffled_images, shuffled_data

def split_datasets(shuffled_images, shuffled_data, train_proportion=0.8):
	#from keras.utils import np_utils
	images_train = shuffled_images[0:math.floor(train_proportion*len(shuffled_data))]
	images_test = shuffled_images[(math.floor(train_proportion*len(shuffled_data))+1):len(shuffled_data)]
	data_train = shuffled_data[0:math.floor(train_proportion*len(shuffled_data))]
	data_test = shuffled_data[(math.floor(train_proportion*len(shuffled_data))+1):len(shuffled_data)]
	train_labels = data_train['Emotion'].values
	test_label_classes = data_test['Emotion'].values
	
	train_labels = train_labels.astype(np.int)
	test_labels = test_label_classes.astype(np.int)
	fnames_train = data_train['Filename'].values
	fnames_test = data_test['Filename'].values
	Train_labels = np_utils.to_categorical(train_labels, len(np.unique(train_labels)))
	Test_labels = np_utils.to_categorical(test_labels, len(np.unique(test_labels)))

	#del shuffled_images, shuffled_df

	images_train = images_train.astype('float32')
	images_test = images_test.astype('float32')
	images_train /= 255
	images_test /= 255
	return images_train, images_test, Train_labels, Test_labels, test_label_classes, fnames_train, fnames_test

def clean_data_labels(data):
	data['emotion'] = data['emotion'].str.lower()
	data['emotion'] = data['emotion'].replace(to_replace='surprised', value='surprise')
	data['emotion'] = data['emotion'].replace(to_replace='angry', value='anger')
	data['emotion'] = data['emotion'].replace(to_replace='fearful', value='fear')
	data['emotion'] = data['emotion'].replace(to_replace='disgusted', value='disgust')
	#####
	data['emotion'] = data['emotion'].replace(to_replace='surprise', value='fear')
	data['emotion'] = data['emotion'].replace(to_replace='contempt', value='anger')
	data['emotion'] = data['emotion'].replace(to_replace='disgust', value='anger')
	#####
	data['emotion'] = data['emotion'].replace(to_replace='fear', value='4')
	data['emotion'] = data['emotion'].replace(to_replace='sadness', value='3')
	data['emotion'] = data['emotion'].replace(to_replace='anger', value='2')
	data['emotion'] = data['emotion'].replace(to_replace='neutral', value='1')
	data['emotion'] = data['emotion'].replace(to_replace='happiness', value='0')
	return data

def reshape_image_array(image_array):
	image_array = image_array[..., 0]
	image_array = np.dstack(image_array)
	image_array = np.rollaxis(image_array, -1)
	image_array = image_array[:, np.newaxis, :, :]
	return image_array

def import_dep_packages():
	global keras, cifar10, ImageDataGenerator, load_model, accuracy_score, confusion_matrix, cohen_kappa_score, Sequential, Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D, SGD, np_utils, l2, activity_l2, StratifiedKFold
	import keras
	from keras.datasets import cifar10
	from keras.preprocessing.image import ImageDataGenerator
	from keras.models import Sequential, load_model
	from keras.layers import Dense, Dropout, Activation, Flatten
	from keras.layers import Convolution2D, MaxPooling2D
	from keras.optimizers import SGD
	from keras.utils import np_utils
	from keras.regularizers import l2, activity_l2
	from sklearn.cross_validation import StratifiedKFold
	from sklearn.metrics import cohen_kappa_score, confusion_matrix, accuracy_score

def create_model(num_conv_filters_layer1, num_conv_kernel_rows, num_conv_kernel_cols, num_conv_filters_layer2):
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
	model.add(Dense(128)) #model.add(Dense(512))
	#model.add(Dropout(0.5)) #0.5
	model.add(Activation(act))
	model.add(Dense(64)) #model.add(Dense(512)) #added
	#model.add(Dropout(0.5)) #0.5 #added
	model.add(Activation(act)) #added
	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))
	return model

def output_metrics(model, images_test, test_label_classes):
	print('Predicting labels for test data...')
	preds = model.predict_classes(images_test)
	test_label_classes = test_label_classes.astype(np.int)
	#np.save('C:\\ML\\IS622\\test_labels.npy', test_labels)
	#np.save('C:\\ML\\IS622\\preds_classes.npy', preds)
	#test_labels = np.load('C:\\ML\\IS622\\test_labels_classes.npy')
	print('Confusion Matrix: ')
	print(confusion_matrix(test_label_classes, preds))
	print('Kappa score: ', cohen_kappa_score(test_label_classes, preds))
	print('Accuracy score: ', accuracy_score(test_label_classes, preds))

if __name__ == '__main__':
	num_conv_filters_layer1 = 48
	num_conv_filters_layer2 = 32
	num_conv_kernel_rows, num_conv_kernel_cols = 3, 3
	batch_size = 64 #originally 32
	nb_classes = 5
	nb_epoch = 100
	data_augmentation = True
	test_proportion = 0.25
	train_proportion = 1 - test_proportion
	img_rows, img_cols = 350, 350 #32, 32
	img_channels = 1 #3
	learning_rate = 0.001 #0.001
	augmentdata1 = False
	n_folds = 10
	data_file = 'C:\\Users\\Aadi\\Documents\\GitHub\\facial_expressions\\data\\legend.csv'
	img_loc = 'C:\\Users\\Aadi\\Documents\\GitHub\\facial_expressions\\images\\'
	print('Reading CSV Data...')
	data_labels = pd.read_csv(data_file)
	data_labels = clean_data_labels(data_labels)
	print('Loading Images...')
	shuffled_images, shuffled_data = load_data(data_labels, img_loc)
	print('Loading Images Complete.')
	import_dep_packages()
	print('Splitting Datasets')
	images_train, images_test, train_labels, test_labels, test_label_classes, fnames_train, fnames_test = split_datasets(shuffled_images, shuffled_data)
	model = create_model(36, 3, 3, 24)
	sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
	print('Compiling model...')
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
	datagen = ImageDataGenerator(
	    featurewise_std_normalization=False,
	    rotation_range = 20,
	    width_shift_range = 0.10,
	    height_shift_range = 0.10,
	    shear_range = 0.1,
	    zoom_range = 0.1,
	    horizontal_flip = True)
	#datagen.fit(images_train, augment=True, rounds=10)
	print('Training Model: ')
	model.fit_generator(datagen.flow(images_train, train_labels,
	               batch_size=batch_size),
	               samples_per_epoch=1300, #1300 images x 100 epochs will give model ten times the initial data size of images
	               nb_epoch=nb_epoch,
	               validation_data=(images_test, test_labels))
	model.save('C:\\ML\\IS622\\nn1.h5')
	#model = load_model('C:\\ML\\IS622\\nn1.h5')
	output_metrics(model, images_test, test_label_classes)

