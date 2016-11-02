Team Member:
	Ye Zhang (zhangyeus@gmail.com)
	Wei Luo (luoweiforever@gmail.com)



Dependencies:
	Tensorflow backend 1.10
	Keras (licenced by MIT)
	nvidia cuda 7.5
	python 2 or 3
	
DATASET:
	Training data: dataset 2-1 offered by Udacity
	Predict data: dataset 2-2 offered by Udacity
	Note: We excluded the last 64 seconds in the training data because during that period the car stoped and the steering angle kept the same.



Models
	CNN:
	layer1: Conv2D 64*3*3
		Dropout 50%
	layer2: Conv2D 64*3*3
		Dropout 50%
		MaxPool 2*2
	layer3: Conv2D 128*3*3
		Dropout 50%
		MaxPool 2*2
	layer4: Conv2D 128*2*2
		Dropout 50%
		MaxPool 2*2
	layer5:  - flatten
	layer6: Fully connected, dense 4096
		Dropout 50%
	layer7: Fully connected, dense 1024
		Dropout 50%
	layer8: Fully connect, dense 1024
		Output 

	We added a restriction to the output where the absolute value of predicted steering angle cannot exceed 3*pi.  



How to run:
	1. Run main.py (which is training model using dataset 2-1 + predicting using dataset 2-2)
	#################### Command to run ####################################################
	python main.py --dataset /data/output --test-dataset /data/output  --nb-epoch 20 --resized-image-width 60 --resized-image-height 80 --test-batch_size 16

	2. Run evaluation model (which uses 70% for training and 30% for testing)
	#################### Command to run ####################################################
	python evaluation_model.py --dataset /data/output  --nb-epoch 2 --resized-image-width 60 --resized-image-height 80

	3. evaluate model with existed cnn_weights.hdf5
	#################### Command to run ####################################################
	python evaluation_model_load_weights.py --dataset /data/output  --nb-epoch 2 --resized-image-width 60 --resized-image-height 80 --weights-path=/data/cnn_weights.hdf5




Reference:
	1. Nvidia end-to-end deep learn for self-driving cars
	2. Used materials from https://github.com/dolaameng/udacity-SDC-baseline
