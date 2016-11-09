'''
Construct data generator to load images and steerings
from dataset folder and provide feeds to Keras models.
Dataset (images and steerings) can be generated by reading
rosbag files. The generator assumes the dataset folder has
the following structure.
```
camera.csv  center/  left/  right/  steering.csv
```
'''
from __future__ import print_function
import numpy as np
from collections import defaultdict
from os import path
from scipy.misc import imread, imresize

#from keras.applications import vgg16
from keras import backend as K


def read_steerings(steering_log, time_scale):
    steerings = defaultdict(list)
    with open(steering_log) as f:
        for line in f.readlines()[1:]:
            fields = line.split(",")
            nanosecond, angle = int(fields[1]), float(fields[2])
            timestamp = nanosecond / time_scale
            steerings[timestamp].append(angle)
    return steerings

def read_image_stamps(image_log, camera, time_scale):
    timestamps = defaultdict(list)
    with open(image_log) as f:
        for line in f.readlines()[1:]:
            if camera not in line:
                continue
            fields = line.split(",")
            nanosecond = int(fields[1]) 
            timestamp = nanosecond / time_scale
            timestamps[timestamp].append(nanosecond)
    return timestamps

def read_images(image_folder, camera, ids, image_size):
    prefix = path.join(image_folder, camera)
    imgs = []
    for id in ids:
        img = imread(path.join(prefix, '%d.jpg' % id))
        img = imresize(img, size=image_size)
        imgs.append(img)
    img_block = np.stack(imgs, axis=0)
    if K.image_dim_ordering() == 'th':
        img_block = np.transpose(img_block, axes = (0, 3, 1, 2))
    return img_block

def normalize_input(x):
	return x / 255.

def exact_output(y):
    return y

def normalize_output(y):
	return y / 5. 

def data_generator(steerings, image_stamps, image_folder, 
                   batch_size=8, camera='center', fps=10, image_size=0.5,
                   timestamp_start=None, timestamp_end=None, shuffle=True):
    # setup
    minmax = lambda xs: (min(xs), max(xs))
    
    # statistics report
    print('timestamp range for all steerings: %d, %d' % minmax(steerings.keys()))
    print('timestamp range for all images: %d, %d' % minmax(image_stamps.keys()))
    print('min and max # of steerings per time unit: %d, %d' % minmax(map(len, steerings.values())))
    print('min and max # of images per time unit: %d, %d' % minmax(map(len, image_stamps.values())))
    
    # generate images and steerings within one time unit.
    # mean steering will be used for mulitple steering angels within the unit.
    start = max(min(steerings.keys()), min(image_stamps.keys()))
    if timestamp_start:
        start = max(start, timestamp_start)
    end = min(max(steerings.keys()), max(image_stamps.keys()))
    if timestamp_end:
        end = min(end, timestamp_end)
    print("data from timestamp %d to %d" % (start, end))
    
    i = start
    x_buffer, y_buffer, buffer_size = [], [], 0
    while True:
        if i > end: 
            i = start
        # find next batch of images and steering
        images = read_images(image_folder, camera, image_stamps[i], image_size)
        # use mean angel with a time unit
        angels = np.repeat([np.mean(steerings[i])], images.shape[0])
        x_buffer.append(images)
        y_buffer.append(angels)
        buffer_size += images.shape[0]
        if buffer_size >= batch_size:
            indx = range(buffer_size)
            np.random.shuffle(indx)
            x = np.concatenate(x_buffer, axis=0)[indx[:batch_size], ...]
            y = np.concatenate(y_buffer, axis=0)[indx[:batch_size], ...]
            x_buffer, y_buffer, buffer_size = [], [], 0
            yield normalize_input(x.astype(np.float32)), exact_output(y)
        if shuffle:
            i = np.random.randint(start, end)
        else:
            i += 1
