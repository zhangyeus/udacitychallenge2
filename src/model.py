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

	img_input = Input(input_shape)

	x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(img_input)
	x = Dropout(0.5)(x)
	x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(x)
	x = Dropout(0.5)(x)
	x = MaxPooling2D((2, 2), strides=(2, 2))(x)

	x = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(x)
	x = Dropout(0.5)(x)
	# it doesn't fit in my GPU
	# x = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(x)
	# x = Dropout(0.5)(x)
	x = MaxPooling2D((2, 2), strides=(2, 2))(x)

	y = Flatten()(x)
	y = Dense(1024, activation='relu')(y)
	y = Dropout(.5)(y)
	y = Dense(1024, activation='relu')(y)
	y = Dropout(.5)(y)
	y = Dense(1)(y)

	model = Model(input=img_input, output=y)
	model.compile(optimizer=Adam(lr=1e-4), loss = 'mse')
	return model