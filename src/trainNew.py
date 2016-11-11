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
                                 #timestamp_end=14751877262-640,
                                 shuffle = True)
	model_saver = ModelCheckpoint(filepath="cnn_weights.hdf5", verbose=1, save_best_only=False)
	hist = model_cnn.fit_generator(train_generator, samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch,
						callbacks=[model_saver])
	print(hist.history)
	print('cnn_model successfully trained...')	
	##########################################################################################################

def train_model_new(steering_log,model_cnn,image_log,image_folder,camera,batch_size,image_size,nb_epoch,fps=10):
	time_scale = int(1e9) / fps
	    # read steering and image log
	steerings = read_interpolate_new(steering_log, camera,	time_scale)
	image_stamps = read_image_stamps(steering_log, camera, time_scale)

	samples_per_epoch = len(image_stamps.keys())*fps/2
	train_generator = data_generator_new(steerings=steerings,
                                 image_stamps=image_stamps, 
                                 image_folder=image_folder,
                                 camera=camera,
                                 batch_size=batch_size,
                                 image_size=image_size,
                                 #timestamp_end=14751877262-640,
                                 shuffle = True)
	model_saver = ModelCheckpoint(filepath="cnn_weights.hdf5", verbose=1, save_best_only=False)
	hist = model_cnn.fit_generator(train_generator, samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch,
						callbacks=[model_saver])
	print(hist.history)
	print('cnn_model successfully trained...')	
	##################################################################################################
def training_new_model(steering_log,model_cnn,image_log,image_folder,camera,batch_size,image_size,nb_epoch):
	i =1
	while(i<=20):
		print("This is new model training %d/20"%i)
		train_model_new(steering_log=steering_log[i-1],
			model_cnn=model_cnn,
			image_log=image_log[i-1],
			image_folder=image_folder[i-1],
			camera=camera,
			batch_size=batch_size,
			image_size=image_size,
			nb_epoch=nb_epoch)
		i+=1





	##################################################################################################

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
			xx=normalize_input(test_x.astype(np.float32))
			buffer_size,images_buffer=0,[]
			test_y = model_cnn.predict(xx)
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
		xx=normalize_input(test_x.astype(np.float32))
		test_y = model_cnn.predict(xx)
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
	parser.add_argument('--dataset1', type=str, help='dataset folder with csv and image folders')
	parser.add_argument('--dataset2', type=str, help='dataset folder with csv and image folders')
	parser.add_argument('--dataset3', type=str, default='',help='dataset folder with csv and image folders')
	parser.add_argument('--test-dataset', type=str, help='dataset folder with csv and image folders')
	parser.add_argument('--weights_path1', type=str, default='',help='# of trained model weights path')
	parser.add_argument('--weights_path2', type=str, default='',help='# of trained model weights path')

	parser.add_argument('--resized-image-width', type=int, default=60, help='image resizing')
	parser.add_argument('--resized-image-height', type=int,default=80, help='image resizing')
	parser.add_argument('--nb-epoch', type=int, default=4, help='# of training epoch')
	parser.add_argument('--trainNum', type=int, default=1, help='# of training total')
	parser.add_argument('--camera', type=str, default='center', help='camera to use, default is center')
	parser.add_argument('--batch_size', type=int, default=20, help='training batch size')
	parser.add_argument('--test-batch_size', type=int, default=60, help='testing batch size')
	args = parser.parse_args()

	dataset_path1 = args.dataset1
	dataset_path2 = args.dataset2
	test_dataset_path = args.test_dataset
	image_size = (args.resized_image_width, args.resized_image_height)
	camera = args.camera
	batch_size = args.batch_size
	test_batch_size = args.test_batch_size
	nb_epoch = args.nb_epoch
	weights_path1 = args.weights_path1
	weights_path2 = args.weights_path2
	train_Num = args.trainNum

	# build model and train it
	steering_log1 = path.join(dataset_path1, 'steering.csv')
	steering_log2 = path.join(dataset_path2, 'steering.csv')
	dataset_path3 = []
	steering_log3 =[]
	jj=1
	while(jj<=20):
		dataset_path3.append(path.join(args.dataset3, str(jj)))
		steering_log3.append(path.join(dataset_path3[jj-1], 'interpolated.csv'))
		jj+=1
	image_log1 = path.join(dataset_path1, 'camera.csv')
	image_log2 = path.join(dataset_path2, 'camera.csv')
	image_log3 = steering_log3
	test_image_log = path.join(test_dataset_path, 'camera.csv')

	camera_images1 = dataset_path1
	camera_images2 = dataset_path2
	camera_images3 = dataset_path3
	test_camera_images = test_dataset_path

	model_cnn = build_cnn(image_size)
	print('model  build successful...')
	K.set_value(model_cnn.optimizer.lr, 4e-4)
	learnRate_ini=K.get_value(model_cnn.optimizer.lr)
	print("Initial learning rate: ", learnRate_ini)
	learnRate=learnRate_ini
	###############################################################
	ii=1
	while (ii <= train_Num ):
		learnRate*=0.5
		#########train new model##########################################
		nb_epoch=1
		training_new_model(steering_log = steering_log3,
		image_log = image_log3,
		image_folder = camera_images3,
		camera = camera,
		batch_size = batch_size,
		model_cnn = model_cnn,
		nb_epoch = nb_epoch,
		image_size =image_size
		)
		print('part2 training done 1/3')
		time.sleep(2)
		#########dataset 1 ############################################

		K.set_value(model_cnn.optimizer.lr, learnRate)
		learnRate=K.get_value(model_cnn.optimizer.lr)
		nb_epoch=2
		print("This is current learn rate: ",learnRate)
		print("start: ",ii,'/',train_Num)
		print("batchsize: ",batch_size)
		train_model(steering_log = steering_log1,
		image_log = image_log1,
		image_folder = camera_images1,
		camera = camera,
		batch_size = batch_size,
		model_cnn = model_cnn,
		nb_epoch = nb_epoch,
		image_size =image_size
		)
		print('part1 training done 2/3')
		time.sleep(2)
		#load_trained_model(model=model_cnn,weights_path=weights_path2)
		#print('dataset weight 2 loaded successfully')
		####################data set 2 #################################
		nb_epoch=16
		train_model(steering_log = steering_log2,
		image_log = image_log2,
		image_folder = camera_images2,
		camera = camera,
		batch_size = batch_size,
		model_cnn = model_cnn,
		nb_epoch = nb_epoch,
		image_size =image_size
		)
		print('part2 training done 3/3')
		time.sleep(2)


		


		ii=ii+1
	


	################################################################
	#begin to train model

	# test 
	test_predict(test_image_log=test_image_log,
		test_image_folder=test_camera_images,
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
