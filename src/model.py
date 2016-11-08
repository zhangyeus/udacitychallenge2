from keras.applications import vgg16
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Dropout, Input, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras import backend as K

def build_cnn(image_size=None):
	image_size = image_size or (60, 80)
	if K.image_dim_ordering() == 'th':
	    input_shape = (3,) + image_size
	else:
	    input_shape = image_size + (3, )

	#img_input = Input(input_shape)
	inputShape=input_shape
	############################ Nvidia ###################################

	############################# 4 conv  ##################################
	
	model=Sequential()
	model.add(Convolution2D(64,3,3, activation='relu', border_mode='same', input_shape=inputShape))
	model.add(Dropout(0.5))
	model.add(Convolution2D(64,3,3, activation='relu', border_mode='same'))
	model.add(Dropout(0.5))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))
	model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
	model.add(Dropout(0.5))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))
	#model.add(Convolution2D(128, 2, 2, activation='relu', border_mode='same'))
	#model.add(Dropout(0.5))
	#model.add(MaxPooling2D((2, 2), strides=(2, 2)))
	##
	model.add(Flatten())
	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1))
	
	##########################$ 6 conv ############################
	'''
	model=Sequential()
	model.add(Convolution2D(64,3,3, activation='relu', border_mode='same', input_shape=inputShape))
	model.add(Dropout(0.5))
	model.add(Convolution2D(64,3,3, activation='relu', border_mode='same'))
	model.add(Dropout(0.5))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))
	model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
	model.add(Dropout(0.5))
	model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
	model.add(Dropout(0.5))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))
	#model.add(MaxPooling2D((2, 2), strides=(2, 2)))
	#model.add(Convolution2D(48, 2, 2, activation='relu', border_mode='same'))
	#model.add(Dropout(0.5))
	#model.add(MaxPooling2D((2, 2), strides=(2, 2)))
	##
	model.add(Flatten())
	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1024, activation='relu'))
	#model.add(Dropout(0.5))
	model.add(Dense(1))
	'''
	###############################################################

	
	#model.compile(optimizer=Adam(lr=1e-4), loss = 'mse')
	#model.compile(optimizer=Adam(lr=0.006302), loss = 'mse')
	model.compile(optimizer=Adam(lr=1e-4), loss = 'mse')
	return model