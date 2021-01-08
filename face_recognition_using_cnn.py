#face recognition using opencv and cnn with faces of elon musk and jeff bezos

# -*- coding: utf-8 -*-
'''By Ajay'''

import cv2 as cv
import numpy as np
import os
import threading as th
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Dropout,Activation,Flatten

#loading required folder containing faces
file1=os.listdir(r'F:\celeb pics\jeff_bezos')
file2=os.listdir(r'F:\celeb pics\elon_musk')

labels=['jeff_bezos','elon_musk']
jeff=[]
musk=[]

#making functions to detect and crop faces
def jeffb():
    for i in file1:
        img1=cv.imread (os.path.join(r'F:\celeb pics\jeff_bezos',i))
        #loading haarcascade xml file
        cascade1 = cv.CascadeClassifier('C:\\Users\\Ajay\\Anaconda3\\Library\\etc\\haarcascades\\haarcascade_frontalface_default.xml')
        gray_image1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    
        # Applying the haar classifier to detect faces
        face_coord1 = cascade1.detectMultiScale(gray_image1, scaleFactor=1.1, minNeighbors=7)
    
        for (x, y, w, h) in face_coord1:
            cv.rectangle(img1, (x, y), (x+w, y+h), (0, 255, 0), 3)
            imgg1=img1[y:y+h,x:x+w]
            imgg1=cv.resize(imgg1,(60,60))
            imgg1=cv.cvtColor(imgg1,cv.COLOR_BGR2GRAY)
            jeff.append(imgg1)
        cv.namedWindow('detected1',cv.WINDOW_NORMAL)
        cv.imshow('detected1', img1)
        k=cv.waitKey(100)
        if k== 27:
            break
    cv.destroyAllWindows() 
    for ij in range(10):
        cv.imshow('sdvs',jeff[ij])
        print(jeff[ij].shape)
        cv.waitKey(100)
        cv.destroyAllWindows()

def elonm():
    for j in file2:
        img2=cv.imread(os.path.join(r'F:\celeb pics\elon_musk',j))
        cascade2 = cv.CascadeClassifier('C:\\Users\\Ajay\\Anaconda3\\Library\\etc\\haarcascades\\haarcascade_frontalface_default.xml')
        gray_image2 = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
    
        # Applying the haarcascade classifier to detect faces
        face_coord2 = cascade2.detectMultiScale(gray_image2, scaleFactor=1.1, minNeighbors=7)
    
        for (m, n, o, p) in face_coord2:
            cv.rectangle(img2, (m, n), (m+o, n+p), (0, 255, 0), 3)
            imgg2=img2[n:n+p,m:m+o]
            imgg2=cv.resize(imgg2,(60,60))
            imgg2=cv.cvtColor(imgg2,cv.COLOR_BGR2GRAY)
            musk.append(imgg2)
        cv.namedWindow('detected2',cv.WINDOW_NORMAL)
        cv.imshow('detected2', img2)
        k=cv.waitKey(100)
        if k== 27:
            break
    cv.destroyAllWindows()
    for kl in range(10):
        cv.imshow('sdvs',musk[kl])
        print(musk[kl].shape)
        cv.waitKey(100)
        cv.destroyAllWindows()
      
#using multithreading to detect faces simultanoeusly
t1=th.Thread(target=jeffb)
t2=th.Thread(target=elonm)
t1.start()
t2.start()
t1.join()
t2.join()

#creating training dataset with images and their labels
train_data=[]

for label in labels:
    if label=='jeff_bezos':
        for img in jeff:
            nlabel=labels.index(label)
            train_data.append([img,nlabel])
    elif label=='elon_musk':
        for img in musk:
            nlabel=labels.index(label)
            train_data.append([img,nlabel])
  
#shuffling training data
random.shuffle(train_data)
train_data=np.array(train_data)
train_data.shape

#separating images(feature variable) and labels(target variable) for training
pics=[]
la=[]
for p,l in train_data:
    pics.append(p)
    la.append(l)
pics=np.array(pics)
la=np.array(la)
pics=pics.reshape(-1,60,60,1)    
pics=pics/255
pics.shape
print(pics.shape[1:])

#training the model using CNN to recognize face
k_mod=Sequential()
k_mod.add(Conv2D(40,(3,3),input_shape=pics.shape[1:]))
k_mod.add(Activation('relu'))
k_mod.add(MaxPooling2D(pool_size=(2,2)))
  
k_mod.add(Conv2D(40,(3,3)))
k_mod.add(Activation('relu'))
k_mod.add(MaxPooling2D(pool_size=(2,2))) 

k_mod.add(Flatten())
k_mod.add(Dense(50,activation='relu'))
k_mod.add(Dense(1,activation='sigmoid'))

#compiling the model
k_mod.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

k_mod.summary()

#fitting the model
k_mod.fit(pics,la,epochs=50,shuffle=True)

#testing the model on an image
test_img=cv.imread(r'C:\Users\Ajay\Downloads\sample.jpg')
cascade1 = cv.CascadeClassifier('C:\\Users\\Ajay\\Anaconda3\\Library\\etc\\haarcascades\\haarcascade_frontalface_default.xml')
gray_test = cv.cvtColor(test_img, cv.COLOR_BGR2GRAY)

# Applying the haar classifier to detect face 
test_coord1 = cascade1.detectMultiScale(gray_test, scaleFactor=1.1, minNeighbors=7)

for (x, y, w, h) in test_coord1:
    cv.rectangle(test_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
    imgg1=test_img[y:y+h,x:x+w]
    imgg1=cv.resize(imgg1,(60,60))
    imgg1=cv.cvtColor(imgg1,cv.COLOR_BGR2GRAY)
imgg1=imgg1.reshape(-1,60,60,1)
imgg1=imgg1/255

cv.imshow('fffh',imgg1.reshape(60,60))
cv.waitKey(0)
cv.destroyAllWindows()
pre=k_mod.predict(imgg1)
print(pre)
pre=pre[0][0]

#printing the probability
print(pre)

#printing recognized output
if pre<=0.4:
    print('jeff_bezos')
elif pre>0.4:
    print('elon_musk')
