#face recognition on live video using cnn and opencv dnn

# -*- coding: utf-8 -*-
'''By Ajay'''

#importing required libraries
import numpy as np
import cv2 as cv
import keyboard as kb
import os
import threading
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,MaxPooling2D,Conv2D,Flatten
import random

#loading folder containing random faces to deal with unknown faces
file1=os.listdir(r'F:\face_pics\unknown')

#defining labels
labels=['ajay','unknown']

#loading opencv dnn caffe model
modelFile = "C:\\Users\\Ajay\\models\\res10_300x300_ssd_iter_140000.caffemodel"
configFile = "C:\\Users\\Ajay\\models\\deploy.prototxt"
net = cv.dnn.readNetFromCaffe(configFile, modelFile)

#Training part
#starting camera
capt=cv.VideoCapture(0)
my_im=[]
def my_images():
    try:

        var1=1
        while True:
            _,img=capt.read()
            h, w = img.shape[:2]
            blob = cv.dnn.blobFromImage(cv.resize(img, (250, 250)), 1.0,
            (250, 250), (104.0, 117.0, 123.0))
            net.setInput(blob)
            faces = net.forward()
            print(faces.shape[2])
            #drawing boxes on faces in image
            for i in range(faces.shape[2]):
                    confidence = faces[0, 0, i, 2]
                    print(confidence)
                    print('---------------')
                    if confidence > 0.5:
                        box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (x, y, x1, y1) = box.astype("int")
                        cv.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 2)
            gray_image=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
            gray_image=gray_image[y:y1,x:x1]
            gray_image=cv.resize(gray_image,(50,50),cv.INTER_AREA)
            #saving preprocessed images of my face to a folder and an empty list declared above
            if kb.is_pressed('ctrl')==True:
                my_im.append(gray_image)
                os.chdir('F:\\face_pics\\ajaysagar')
                cv.imwrite(str(var1)+'.jpg',gray_image)
                var1+=1
            cv.imshow('detected', img)
            k=cv.waitKey(1)
            if k== 27:
                break
    except:
        pass
    capt.release()
    cv.destroyAllWindows() 

#preprocessing random faces and saving them to'unknown' empty list    
unknown=[]
def unknwn():
    for i in file1:
        img1=cv.imread (os.path.join(r'F:\face_pics\unknown',i))
        cascade1 = cv.CascadeClassifier('C:\\Users\\Ajay\\Anaconda3\\Library\\etc\\haarcascades\\haarcascade_frontalface_default.xml')
        gray_image1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    
        # Applying the haar classifier to detect faces
        face_coord1 = cascade1.detectMultiScale(gray_image1, scaleFactor=1.1, minNeighbors=12)
    
        for (x, y, w, h) in face_coord1:
            cv.rectangle(img1, (x, y), (x+w, y+h), (0, 255, 0), 3)
        imgg1=img1[y:y+h,x:x+w]
        imgg1=cv.resize(imgg1,(50,50))
        imgg1=cv.cvtColor(imgg1,cv.COLOR_BGR2GRAY)
        unknown.append(imgg1)
        cv.imshow('g',imgg1)
        cv.waitKey(20)
        cv.destroyAllWindows()
#running the function simultaneously using multithreading
th1=threading.Thread(target=my_images)
th2=threading.Thread(target=unknwn)
th1.start()
th2.start()
th1.join()
th2.join()

#preparing images and labels
cnn_train=[]
for label in labels:
    if label=='ajay':
        for img in my_im:
            nlabel=labels.index(label)
            cnn_train.append([img,nlabel])
    elif label=='unknown':
        for img in unknown:
            nlabel=labels.index(label)
            cnn_train.append([img,nlabel])
 
random.shuffle(cnn_train)
cnn_train=np.array(cnn_train)
print(cnn_train.shape)

#separating images(feature variable) and labels(target variable) for training
pics=[]
la=[]
for p,l in cnn_train:
    pics.append(p)
    la.append(l)
pics=np.array(pics)
la=np.array(la)
pics=pics.reshape(-1,50,50,1)    
pics=pics/255

cv.imshow('gh',pics[10])
cv.waitKey(0)
cv.destroyAllWindows()

#training the model using CNN to recognize face
k_mod=Sequential()
k_mod.add(Conv2D(50,(3,3),activation='relu',input_shape=pics.shape[1:]))
k_mod.add(MaxPooling2D(pool_size=(2,2)))
  
k_mod.add(Conv2D(50,(3,3),activation='relu'))
k_mod.add(MaxPooling2D(pool_size=(2,2))) 

k_mod.add(Flatten())
k_mod.add(Dense(50,activation='relu'))
k_mod.add(Dense(1,activation='sigmoid'))

#compiling the model
k_mod.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

#fitting the model
k_mod.fit(pics,la,epochs=50,shuffle=True)

#saving the trained model
k_mod.save('my_cnn.sav')


#Testing part
#loading the saved model
os.chdir('F:\\face_pics\\ajaysagar')
k_modd=keras.models.load_model('my_cnn.sav')

#starting camera
capt=cv.VideoCapture(0)
while 1:
    _,img=capt.read()
    h, w = img.shape[:2]
    blob = cv.dnn.blobFromImage(cv.resize(img, (250, 250)), 1.0,
    (250, 250), (104.0, 117.0, 123.0))
    net.setInput(blob)
    faces = net.forward()
    print(faces.shape[2])
    for i in range(faces.shape[2]):
            confidence = faces[0, 0, i, 2]
            if confidence > 0.5:
                box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                cv.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 2)
                gray_image=img[y:y1,x:x1]
                gray_image=cv.cvtColor(gray_image,cv.COLOR_BGR2GRAY)
                gray_image5=cv.resize(gray_image,(50,50),cv.INTER_AREA)
                gray_image6=gray_image5.reshape(-1,50,50,1)
                gray_image6=gray_image6/255
                prediction=k_modd.predict(gray_image6)
                print(prediction)
                #recogninizing the face 
                if prediction[0][0]<=0.6:
                    cv.putText(img,'ajay',(x,y1),cv.FONT_HERSHEY_SIMPLEX,1, (0,255,40), 3)
                else:
                    cv.putText(img,'unknown',(x,y1),cv.FONT_HERSHEY_SIMPLEX,1, (0,255,40), 3)
   
    cv.imshow('detected', img)
    k=cv.waitKey(10)
    if k== 27:
        break
capt.release()
cv.destroyAllWindows()