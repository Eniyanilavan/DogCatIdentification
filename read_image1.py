import tensorflow as tf
import os
from tqdm import tqdm
import cv2
from random import shuffle
import numpy as np
class_name=[]
class_name_val=[]
images_train=[]
image_size = 32
true_label = []
image_data = []
image_val_data = []
images_val=[]
true_val_label = []
def image_class_train(base_dir = "photos"):  # to find the images'  path
        global num_class
        # global labels
        # global  in_label
        sub_dirs = [x[0] for x in tf.gfile.Walk(base_dir)]  # to walk in the specified directory
        num_class = len(sub_dirs) - 1  # to find number of categories

        # print(sub_dirs)
        la = True
        for sub_dir in sub_dirs:
            if la == True:
                la = False
                continue
            # print(sub_dir)
            sub_dir = sub_dir.split("\\")
            # print(sub_dir)
            class_name.append(sub_dir[1])


def image_class_val(base_dir="validation"):  # to find the images'  path
    global num_class_val
    # global labels
    # global  in_label
    sub_dirs = [x[0] for x in tf.gfile.Walk(base_dir)]  # to walk in the specified directory
    num_class = len(sub_dirs) - 1  # to find number of categories

    # print(sub_dirs)
    la = True
    for sub_dir in sub_dirs:
        if la == True:
            la = False
            continue
        # print(sub_dir)
        sub_dir = sub_dir.split("\\")
        # print(sub_dir)
        class_name_val.append(sub_dir[1])
def train_image_data(base_dir = "photos"):
    i = 0
    paths = []
    image_class_train(base_dir=base_dir)
    global images_train
    global true_label
    global image_data
    for classe in class_name:
        path_x = []
        label = [0. for _ in range(num_class)]
        path = os.path.join(base_dir, classe, '*')
        paths.extend(tf.gfile.Glob(path))
        path_x.extend(tf.gfile.Glob(path))
        label[i] = 1.
        i += 1
        for _ in range(len(path_x)):
            true_label.append(label)
        del label
        del path_x
    true_label = np.array(true_label)
    # print(paths)
    # print(true_label)
    for path in tqdm(paths):
        image = cv2.imread(path,0)
        image = cv2.resize(image, (image_size, image_size),cv2.INTER_LINEAR)
        image = image[:, :, np.newaxis]
        images_train.append(image)
    images_train = np.array(images_train, dtype=np.float32)
    # images = images.astype('float32')
    images_train = np.multiply(images_train, 1.0 / 225.0)
    # print(images,true_label)
    # cv2.imshow('image', images_train[0])
    # cv2.waitKey(0)
    # print(images_train)
    for i in tqdm(range(0,len(images_train))):
        image_data.append([images_train[i],true_label[i]])
    # print(image_data)

    return image_data,num_class

def val_image_data(base_dir="validation"):
    i = 0
    paths = []
    image_class_val(base_dir=base_dir)
    global images_val
    global true_val_label
    global image_data
    for classe in class_name_val:
        path_x = []
        label_val = [0. for _ in range(num_class)]
        path = os.path.join(base_dir, classe, '*')
        paths.extend(tf.gfile.Glob(path))
        path_x.extend(tf.gfile.Glob(path))
        label_val[i] = 1.
        i += 1
        for _ in range(len(path_x)):
            true_val_label.append(label_val)
        del label_val
        del path_x
    true_val_label = np.array(true_val_label)
    # print(paths)
    # print(true_label)
    for path in tqdm(paths):
        image = cv2.imread(path,0)
        image = cv2.resize(image, (image_size, image_size),cv2.INTER_LINEAR)
        image = image[:, :, np.newaxis]
        images_val.append(image)
    images_val = np.array(images_val, dtype=np.float32)
     # images = images.astype('float32')
    images_val = np.multiply(images_val, 1.0 / 225.0)
    # print(images,true_label)
    # cv2.imshow('image', images_val[0])
    # cv2.waitKey(0)
    for i in tqdm(range(0, len(images_val))):
        image_val_data.append([images_val[i], true_val_label[i]])
    return image_val_data
# train_image_data()
# print(label_name)