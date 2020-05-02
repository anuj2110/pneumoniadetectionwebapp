# -*- coding: utf-8 -*-
"""
Created on Sat May  2 16:19:14 2020

@author: Anuj
"""
# making the imports
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input,Conv2D,MaxPooling2D,Dropout,Flatten,Dense,Activation,BatchNormalization,Add
from tensorflow.keras.models  import Model,Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

#Code for loading training and validation data at the time of training

base_dir = os.getcwd() #getting current directory

target_shape = (224,224) #defining the input shape
train_dir = base_dir+"\\Pneumonia_Detection\\chest_xray\\train" # 
val_dir = base_dir+"\\Pneumonia_Detection\\chest_xray\\val"     # -- Directories for data
test_dir = base_dir+"\\Pneumonia_Detection\\chest_xray\\test"   # 

train_gen = ImageDataGenerator(rescale=1/255.0,
                               horizontal_flip=True,
                               zoom_range=0.2,
                               shear_range=0.2,
                               rotation_range=40) # making the data loader for training data
test_gen = ImageDataGenerator(rescale=1/255.0) # making the data loader for validation data

train_data_gen = train_gen.flow_from_directory(train_dir,
                                               target_shape,
                                               batch_size=16,
                                               class_mode='binary') # function to make iterable object for training
test_data_gen = train_gen.flow_from_directory(test_dir,
                                               target_shape,
                                               batch_size=16,
                                               class_mode='binary') # function to make iterable object for training

# Model Building
def resblock(x,filters,down=False):
    #fucntion to create a resblock for the model
    x_shortcut = x
    f1,f2 = filters
    if down:
        x = Conv2D(f1,kernel_size=(3,3),padding='same',activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(f2,kernel_size=(3,3),padding='same',activation='relu',strides=(2,2))(x)
        x = BatchNormalization()(x)
        x_shorcut = Conv2D(f2,kernel_size=(3,3),activation='relu',padding='same',strides=(2,2))(x_shortcut)
        X = Add()[x,x_shortcut]
        return X
    else:
        x = Conv2D(f1,kernel_size=(3,3),padding='same',activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(f2,kernel_size=(3,3),padding='same',activation='relu')(x)
        x = BatchNormalization()(x)
        x_shorcut = Conv2D(f2,kernel_size=(3,3),activation='relu',padding='same')(x_shortcut)
        X = Add()[x,x_shortcut]
        return X
    
    