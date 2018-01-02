import tflearn as tl
import numpy as np
import os,glob,cv2
import sys,argparse
from read_image import read_valid_image
import tensorflow as tf
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data , dropout, fully_connected
# from tflearn.layers.estimator import  regression

image_data,image_label,num_class,label_name = read_valid_image()
# First, pass the path of the image
dir_path = os.path.dirname(os.path.realpath(__file__))
image_path=sys.argv[1]
filename = dir_path +'/' +image_path
image_size=32
num_channels=1
images = []
# Reading the image using OpenCV
image = cv2.imread(filename,0)

# Resizing the image to our desired size and preprocessing will be done exactly as done during training
image = cv2.resize(image, (image_size, image_size), cv2.INTER_LINEAR)
image = image[:, :, np.newaxis]
images.append(image)
images = np.array(images, dtype=np.uint8)
images = images.astype('float32')
x = np.multiply(images, 1.0/255.0)
# y = np.zeros([1,2])
#The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
# x_batch = images.reshape(1, image_size,image_size,num_channels)
convnet = input_data(shape=[None,32,32,1],name='input')


convnet = conv_2d(convnet,32,2,activation='relu')
convnet = max_pool_2d(convnet,2)


convnet = conv_2d(convnet,64,2,activation='relu')
convnet = max_pool_2d(convnet,2)

convnet = conv_2d(convnet,32,2,activation='relu')
convnet = max_pool_2d(convnet,2)


convnet = conv_2d(convnet,64,2,activation='relu')
convnet = max_pool_2d(convnet,2)

convnet = conv_2d(convnet,32,2,activation='relu')
convnet = max_pool_2d(convnet,2)


convnet = conv_2d(convnet,64,2,activation='relu')
convnet = max_pool_2d(convnet,2)


convnet = fully_connected(convnet, 1024,activation='relu')

convnet = fully_connected(convnet,2,activation='softmax',name="output")
model = tl.DNN(convnet)
model.load('tflearn')
final = model.predict(x)
for i in range(0,len(label_name)):
    print(label_name[i],final[0][i])

# print(final)