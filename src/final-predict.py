from __future__ import print_function

# for docker env
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import csv
import argparse
import numpy as np
from os import path
import os
import os.path
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
                                 #timestamp_end=14751877262-640,
                                 shuffle = True)
	model_saver = ModelCheckpoint(filepath="cnn_weights.hdf5", verbose=1, save_best_only=False)
	model_cnn.fit_generator(train_generator, samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch,
						callbacks=[model_saver])
	print('cnn_model successfully trained...')	

def test_predict_new(test_image_folder,camera,image_size,test_batch_size,model_cnn,loops=None):
	print ('Begin to predict steerings... ')
    # read image log
	#image_stamps = read_image_stamps(test_image_log, camera, time_scale)
	#timestamps = defaultdict(list)
	predict_steer_store,imgindex_store,images_buffer,buffer_size=[],[],[],0
	rootpath=path.join(test_image_folder, camera)
	count=0
	for rt, dirs, files in os.walk(rootpath):
		for ff in files:
			imgs=[]
			fname=os.path.splitext(ff)
			imgindex=fname[0]
			#print("this is fname0:",fname[0])
			#print("this is fname1:",fname[1])
			img = imread(path.join(rootpath, '%s.png' % imgindex))
			img = imresize(img, size=image_size)
			imgs.append(img)
			#print('this is img shape: !!!!!! :', img.shape)
			images = np.stack(imgs, axis=0)
			#print('this is image shape: !!!!!! :', images.shape)
			images_buffer.append(images)
			buffer_size += images.shape[0]
			imgindex_store.append(imgindex)
			#print("imageIndex_store: ",imgindex_store)

			if(loops and count>loops):
				break
			if buffer_size >= test_batch_size:
				count += 1
				print ('Predict per batch size:%s ' %buffer_size)
				test_x = np.concatenate(images_buffer, axis=0)
				xx=normalize_input(test_x.astype(np.float32))
				buffer_size,images_buffer=0,[]
				test_y = model_cnn.predict(xx)
				mask_up=test_y>9.42
				mask_down=test_y<-9.42
				test_y[mask_up]=9.42
				test_y[mask_down]=-9.42
				predict_steer_store.extend(test_y)
				#print("imgindex_store shape", len(imgindex_store))
				#print("predict_steer_store shape",len(predict_steer_store))
	if images_buffer:
		print ('Predict last insufficient batch size images:%s ' %buffer_size)
		test_x = np.concatenate(images_buffer, axis=0)
		xx=normalize_input(test_x.astype(np.float32))
		buffer_size,images_buffer=0,[]
		test_y = model_cnn.predict(xx)
		#test_y = model_cnn.predict(test_x)
		#print ('test_y', test_y)
		mask_up=test_y>9.42
		mask_down=test_y<-9.42
		test_y[mask_up]=9.42
		test_y[mask_down]=-9.42
		predict_steer_store.extend(test_y)
	print("imgindex_store shape", len(imgindex_store))
	print("predict_steer_store shape",len(predict_steer_store))
	predict_steering_dict = dict(zip(imgindex_store,predict_steer_store))
	predict_angle_csv_path = path.join(test_image_folder, 'predict_angle.csv')
	with open (predict_angle_csv_path,'wb') as csv_file:
		writer = csv.writer(csv_file)
		writer.writerow(['frame_id', 'steering_angle'])
		for key, value in predict_steering_dict.items():
			writer.writerow([key, value[0]])



def main():
	# parse arguments
	parser = argparse.ArgumentParser(description="Testing Udacity SDC data")
	parser.add_argument('--test-dataset', type=str, help='dataset folder with csv and image folders')
	parser.add_argument('--weights-path', type=str, help='# of trained model weights path')
	parser.add_argument('--resized-image-width', type=int, default=60, help='image resizing')
	parser.add_argument('--resized-image-height', type=int, default=80, help='image resizing')
	parser.add_argument('--camera', type=str, default='center', help='camera to use, default is center')
	parser.add_argument('--batch_size', type=int, default=8, help='training batch size')
	parser.add_argument('--test-batch_size', type=int, default=20, help='testing batch size')
	args = parser.parse_args()

	test_dataset_path = args.test_dataset
	image_size = (args.resized_image_width, args.resized_image_height)
	camera = args.camera
	test_batch_size = args.test_batch_size
	weights_path = args.weights_path
	test_image_log = path.join(test_dataset_path, 'camera.csv')

	# build model and train it

	test_camera_images = test_dataset_path

	model_cnn = build_cnn(image_size)

	print('model  build successful...')
	print('load weight...')
	load_trained_model(model=model_cnn,weights_path=weights_path)
	print('load weight successfully...')
	#begin to train model

	# test 
	test_predict_new(test_image_folder=test_camera_images,
		camera=camera,image_size=image_size,
		test_batch_size=test_batch_size,
		# loops=3, # for test if enable will be only loops 4
		model_cnn=model_cnn
		)

def load_trained_model(model,weights_path):
	print ('load trained model from %s' %weights_path)
	model.load_weights(weights_path)  

if __name__ == '__main__':
	main()