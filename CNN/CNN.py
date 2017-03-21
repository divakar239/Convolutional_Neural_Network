#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 09:45:02 2017

@author: DK
"""
# Part 1 - Building the CNN

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Part 1: Building CNN
from keras.models import Sequential
from keras.layers import Convolution2D        #add the convolution layers; images are 2D as there is no time so we use 2D
from keras.layers import MaxPooling2D         
from keras.layers import Flatten              #convert all the pools constructed by MaxPooling and Convolution into a single vector which acts as input for the fully connected layers
from keras.layers import Dense                #creates a fully connected network of layers 

#Initialising CNN
classifier=Sequential()

#STEP 1: adding the first(convolutional) layer
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu'))

#STEP 2: MaxPooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Adding another convolutional layer
classifier.add(Convolution2D(32,3,3,activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#STEP 3: Flattening
classifier.add(Flatten())

#Step 4: Full Connection
classifier.add(Dense(output_dim=128,activation='relu'))         #hidden layer
classifier.add(Dense(output_dim=1,activation='sigmoid'))        #output layer

#Compiling CNN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


#Part 2 - Fitting CNN to images
from keras.preprocessing.image import ImageDataGenerator

#Image Augmentation
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)


training_set = train_datagen.flow_from_directory( '/Users/DK/Documents/Machine_Learning/Python-and-R/Machine_Learning_Projects/Convolutional_Neural_Networks/CNN/dataset/training_set',
                                                 target_size=(64,64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory(
        '/Users/DK/Documents/Machine_Learning/Python-and-R/Machine_Learning_Projects/Convolutional_Neural_Networks/CNN/dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        samples_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        num_val_samples=2000)

