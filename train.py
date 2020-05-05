# -*- coding: utf-8 -*-
"""
Created on Sat May  2 16:19:14 2020

@author: Anuj
"""
# making the imports
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input,Conv2D,MaxPooling2D,Dropout,Flatten,Dense,Activation,BatchNormalization,add
from tensorflow.keras.models  import Model,Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
import os

#Code for loading training and validation data at the time of training

base_dir = os.getcwd() #getting current directory

target_shape = (196,196) #defining the input shape
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
    
    merge_input = x
    f1,f2 = filters
    if x.shape[-1] != f2:
        merge_input = Conv2D(f2, (1,1), padding='same', activation='relu', kernel_initializer='he_normal')(x)
    # conv1
    conv1 = Conv2D(f1, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')(x)
    conv1 = BatchNormalization()(conv1)
    # conv2
    conv2 = Conv2D(f2, (3,3), padding='same', activation='linear', kernel_initializer='he_normal')(conv1)
    conv2 = BatchNormalization()(conv2)
    # add filters, assumes filters/channels last
    layer_out = add([conv2, merge_input])
    # activation function
    layer_out = Activation('relu')(layer_out)
    return layer_out

inp = Input(shape=(196,196,3)) #Input layer of CNN
#layer 1 
conv_block1 = resblock(inp,[32,32])
conv_block1 = BatchNormalization()(conv_block1)
conv_block1 = Dropout(0.2)(conv_block1)
conv1 = Conv2D(64,(3,3),padding='same',strides=(2,2),activation='relu')(conv_block1)
conv1 = BatchNormalization()(conv1)
#layer 2
conv_block2 = resblock(conv1,[64,128])
conv_block2 = BatchNormalization()(conv_block2)
conv_block2 = Dropout(0.2)(conv_block2)
conv2 = Conv2D(128,(3,3),padding='same',strides=(2,2),activation='relu')(conv_block2)
conv2 = BatchNormalization()(conv2)
#layer 3
conv_block3 = resblock(conv2,[128,256])
conv_block3 = BatchNormalization()(conv_block3)
conv_block3 = Dropout(0.2)(conv_block3)
conv3 = Conv2D(256,(3,3),padding='same',strides=(4,4),activation='relu')(conv_block3)
conv3 = BatchNormalization()(conv3)
#layer 4
conv_block4 = resblock(conv3,[256,512])
conv_block4 = BatchNormalization()(conv_block4)
conv_block4 = Dropout(0.2)(conv_block4)
conv4 = Conv2D(512,(3,3),padding='same',strides=(4,4),activation='relu')(conv_block4)
conv4 = BatchNormalization()(conv4)
#Flatten
flat = Flatten()(conv4)
#Dense layers
dense_1 = Dense(1024,activation='relu')(flat)
dense_1 = Dropout(0.5)(dense_1)
dense_2 = Dense(512,activation='relu')(dense_1)
dense_2 = Dropout(0.5)(dense_2)
out = Dense(1,activation='sigmoid')(dense_2)

model = Model(inputs=inp,outputs=out)
model.summary()
plot_model(model, to_file='model.png')

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
hist = model.fit_generator(train_data_gen,
        steps_per_epoch=2000,
        epochs=50,
        validation_data=test_data_gen,
        validation_steps=800)

plt.style.use("ggplot")
plt.figure()
plt.plot(hist.history["loss"], label="train_loss")
plt.plot(hist.history["val_loss"], label="val_loss")
plt.plot(hist.history["accuracy"], label="train_acc")
plt.plot(hist.history["val_accuracy"], label="val_acc")
plt.title("Model Training")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig("epochs.png")

