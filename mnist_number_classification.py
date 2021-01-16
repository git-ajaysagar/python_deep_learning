#The following program classifies images of digits(0 to 9) from mnist dataset using neural networks

# -*- coding: utf-8 -*-
'''By Ajay'''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten
import numpy as np
import cv2 as cv
from skimage import img_as_ubyte    

#loading mnist dataset
num_data= keras.datasets.mnist

#dividing it into training and testing
(tx,ty),(tsx,tsy)=num_data.load_data()

tx=tx/255.0
tsx=tsx/255.0

#defining layers of neural network
k_mod=Sequential()
k_mod.add(Flatten(input_shape=(28,28)))
k_mod.add(Dense(250,activation='relu'))
k_mod.add(Dense(10,activation='softmax'))

#compiling the model
k_mod.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#fitting the model
k_mod.fit(tx,ty,epochs=50)

#evaluating the model
ev=k_mod.evaluate(tsx,tsy)
print('This is testing accuracy: ',ev[1])

#testing the trained model on a downloaded image of a digit
imh=cv.imread('C:\\Users\\Ajay\\Downloads\\number 1.jpg')
imh=cv.cvtColor(imh,cv.COLOR_BGR2GRAY)
imh = img_as_ubyte(imh)
_,imh=cv.threshold(imh,120,255,cv.THRESH_BINARY)
imh=cv.resize(imh,(28,28))
imh=imh.reshape(1,28,28)

prediction=k_mod.predict(imh)
print('The output from the model is',np.argmax(prediction))

