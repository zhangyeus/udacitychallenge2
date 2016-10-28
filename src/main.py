from __future__ import print_function

# for docker env
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import csv
import argparse
import numpy as np
from os import path
import time
from data_processer import *
from model import *

from keras.callbacks import ModelCheckpoint

def train_model(steering_log,model_cnn,image_log,image_folder,camera,batch_size,image_size,nb_epoch,fps=10):
	time_scale = int(1e9) / fps
	    # read steering and image log
	steerings = read_steerings(steering_log, time_scale)
	image_stamps = read_image_stamps(image_log, camera, time_scale)

	samples_per_epoch = len(image_stamps.keys())*fps/2
	train_generator = data_generator(steerings=steerings,
                                 image_stamps=image_stamps, 
                                 image_folder=image_folder,
                                 camera=camera,
                                 batch_size=batch_size,
                                 image_size=image_size,
                                 timestamp_end=14751877262-640,
                                 shuffle = True)
	model_saver = ModelCheckpoint(filepath="cnn_weights.hdf5", verbose=1, save_best_only=False)
	model_cnn.fit_generator(train_generator, samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch,
						callbacks=[model_saver])
	print('cnn_model successfully trained...')	

def test_predict(test_image_log,test_image_folder,camera,image_size,test_batch_size,model_cnn,loops=None,fps=10):
	time_scale = int(1e9) / fps
	print ('Begin to predict steerings... ')
    # read image log
	image_stamps = read_image_stamps(test_image_log, camera, time_scale)
	timestamps = defaultdict(list)
	predict_steer_store,timestamps_store,images_buffer,buffer_size=[],[],[],0
	image_stamps_keys = image_stamps.keys()
	count = 0
	for timestamp in image_stamps_keys:
		images = read_images(test_image_folder, camera, image_stamps[timestamp], image_size)
		images_buffer.append(images)
		buffer_size += images.shape[0]
		timestamps_store.extend(image_stamps[timestamp])
		# loops only for test
		if(loops and count>loops):
			break
		if buffer_size >= test_batch_size:
			count += 1
			print ('Predict per batch size:%s ' %buffer_size)
			test_x = np.concatenate(images_buffer, axis=0)
			buffer_size,images_buffer=0,[]
			test_y = model_cnn.predict(test_x)
			mask_up=test_y>9.42
			mask_down=test_y<-9.42
			test_y[mask_up]=9.42
			test_y[mask_down]=-9.42
			#print ('test_y', test_y)
			predict_steer_store.extend(test_y)
	#the last insufficient buffer size images
	if images_buffer:
		print ('Predict last insufficient batch size images:%s ' %buffer_size)
		test_x = np.concatenate(images_buffer, axis=0)
		test_y = model_cnn.predict(test_x)
		#print ('test_y', test_y)
		mask_up=test_y>9.42
		mask_down=test_y<-9.42
		test_y[mask_up]=9.42
		test_y[mask_down]=-9.42
		predict_steer_store.extend(test_y)
	#Write the predict results to csv file
	predict_steering_dict = dict(zip(timestamps_store,predict_steer_store))
	predict_angle_csv_path = path.join(test_image_folder, 'predict_angle.csv')
	with open (predict_angle_csv_path,'wb') as csv_file:
		writer = csv.writer(csv_file)
		writer.writerow(['timestamp', 'predict_steering'])
		for key, value in predict_steering_dict.items():
			writer.writerow([key, value[0]])

def main():
	# parse arguments
	parser = argparse.ArgumentParser(description="Testing Udacity SDC data")
	parser.add_argument('--dataset', type=str, help='dataset folder with csv and image folders')
	parser.add_argument('--test-dataset', type=str, help='dataset folder with csv and image folders')

	parser.add_argument('--resized-image-width', type=int, help='image resizing')
	parser.add_argument('--resized-image-height', type=int, help='image resizing')
	parser.add_argument('--nb-epoch', type=int, help='# of training epoch')
	parser.add_argument('--camera', type=str, default='center', help='camera to use, default is center')
	parser.add_argument('--batch_size', type=int, default=8, help='training batch size')
	parser.add_argument('--test-batch_size', type=int, default=20, help='testing batch size')
	args = parser.parse_args()

	dataset_path = args.dataset
	test_dataset_path = args.test_dataset
	image_size = (args.resized_image_width, args.resized_image_height)
	camera = args.camera
	batch_size = args.batch_size
	test_batch_size = args.test_batch_size
	nb_epoch = args.nb_epoch

	# build model and train it
	steering_log = path.join(dataset_path, 'steering.csv')
	image_log = path.join(dataset_path, 'camera.csv')
	test_image_log = path.join(test_dataset_path, 'camera.csv')

	camera_images = dataset_path
	test_camera_images = test_dataset_path

	model_cnn = build_cnn(image_size)

	print('model  build successful...')
	
	#begin to train model
	train_model(steering_log = steering_log,
	image_log = image_log,
	image_folder = camera_images,
	camera = camera,
	batch_size = batch_size,
	model_cnn = model_cnn,
	nb_epoch = nb_epoch,
	image_size =image_size
	)

	# test 
	test_predict(test_image_log=test_image_log,
		test_image_folder=test_camera_images,
		camera=camera,image_size=image_size,
		test_batch_size=test_batch_size,
		# loops=3, # for test if enable will be only loops 4
		model_cnn=model_cnn
		)  

if __name__ == '__main__':
	main()