# -*- coding: utf-8 -*-
"""
Created on Sat May  2 16:19:14 2020

@author: Anuj
"""
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input,Conv2D,MaxPooling2D,Dropout,Flatten,Dense,Activation,BatchNormalization
from tensorflow.keras.models  import Model,Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

target_shape = (224,224,3)
train_dir = "/Pneumonia_Detection/chest_xray/train"
val_dir = "/Pneumonia_Detection/chest_xray/val"
test_dir = "/Pneumonia_Detection/chest_xray/test"



