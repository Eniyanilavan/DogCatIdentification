import numpy as np
import cv2
from random import shuffle
from read_image1 import train_image_data , val_image_data
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data , dropout, fully_connected
from tflearn.layers.estimator import  regression
# import tflearn.datasets.mnist as mnist
# x, y, test_x, test_y = mnist.load_data(one_hot=True)
# x = x.reshape([-1,28,28,1])
# test_x = test_x.reshape([-1,28,28,1])

#%%

image_data,num_class= train_image_data()
shuffle(image_data)
# x = np.array([i[0]for i in image_data]).reshape(-1,img_size,img_size,1)

# image_val_data = val_image_data()
# shuffle(image_val_data)

# print(image_data)
# print(image_data[1])
x = image_data
test_x = image_data
shuffle(test_x)

input = input_data(shape=[None,32,32,1],name='input')


convnet = conv_2d(input,32,2,activation='relu')
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
# convnet=dropout(convnet,0.8)

convnet = fully_connected(convnet,2,activation='softmax',name="output")
convnet = regression(convnet, optimizer='adam',learning_rate=0.001,loss='categorical_crossentropy',name='targets')

model = tflearn.DNN(convnet,tensorboard_dir='log')

model.fit({'input':[k[0] for k in x]},{'targets':[k[1] for k in x]},n_epoch=20,
          validation_set=({'input':[k[0] for k in test_x]},{'targets':[k[1] for k in test_x]}),
          snapshot_step=500,show_metric=True, run_id='tflearn')

model.save('tflearn')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    tf.train.write_graph(sess.graph_def, "E:\\python\\tensorflow_me\\image recognation",'trainModel.pbtxt')