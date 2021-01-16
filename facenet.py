#multiple face recognition on live video using facenet and opencv dnn

# -*- coding: utf-8 -*-
'''By Ajay'''

#importing required libraries
from tensorflow import keras
import cv2 as cv
import os
import numpy as np
from sklearn.preprocessing import Normalizer 
import random
from sklearn.svm import SVC
import pickle

#loading pretrained facenet model
modell= keras.models.load_model(r'C:\Users\Ajay\Downloads\facenet_keras.h5')

#path to images
fold1=os.listdir(r'F:\facenet_pics\ajay')
fold2=os.listdir(r'F:\facenet_pics\face2')
fold3=os.listdir(r'F:\facenet_pics\face3')

#loading opencv dnn caffe model
modelFile = "C:\\Users\\Ajay\\models\\res10_300x300_ssd_iter_140000.caffemodel"
configFile = "C:\\Users\\Ajay\\models\\deploy.prototxt"
net = cv.dnn.readNetFromCaffe(configFile, modelFile)

#defining labels for faces
labels=['face3','ajay','face2']

#function to process each image and feed it to the facenet model to get the embeddings
def alter(modell,pic):
    pic=pic.astype('float32')
    pic=pic/255.0
    mean=pic.mean()
    std=pic.std()
    pic=(pic-mean)/std
    pic=np.expand_dims(pic,axis=0)
    o_p=modell.predict(pic) 
    return o_p[0]
 
#collecting face embeddings in respective lists
ajay_pics=[]
face2_pics=[]
face3_pics=[]

for f1 in fold1:
    pic=cv.imread(os.path.join(r'F:\facenet_pics\ajay',f1))
    embed=alter(modell, pic)
    ajay_pics.append(embed)

for f2 in fold2:
    pic=cv.imread(os.path.join(r'F:\facenet_pics\face2',f2))
    embed=alter(modell, pic)
    face2_pics.append(embed)                

for f3 in fold3:
    pic=cv.imread(os.path.join(r'F:\facenet_pics\face3',f3))
    embed=alter(modell, pic)
    face3_pics.append(embed)     
    
ajay_pics=np.array(ajay_pics)
face2_pics=np.array(face2_pics)
face3_pics=np.array(face3_pics)

print(ajay_pics.shape)

#normalizing the embeddings
nor=Normalizer(norm='l2')
ajay_pics=nor.transform(ajay_pics)
face2_pics=nor.transform(face2_pics)
face3_pics=nor.transform(face3_pics)

#collecting all the face embeddings and labels in a single list to prepare training data
new_train=[]
for label in labels:
    if label=='face3':
        for u in face3_pics:
            nlabel=labels.index(label)
            new_train.append([u,nlabel]) 
    elif label=='ajay':
        for a in ajay_pics:
            nlabel=labels.index(label)
            new_train.append([a,nlabel])
    elif label=='face2':
        for u in face2_pics:
            nlabel=labels.index(label)
            new_train.append([u,nlabel])
           

print(np.array(new_train).shape)
random.shuffle(new_train)

#separating feature variable(faces) and target vaiable(labels) for training the model
u_pic=[]
u_label=[]
for i1,i2 in new_train:
    u_pic.append(i1)
    u_label.append(i2)

u_pic=np.array(u_pic)
u_label=np.array(u_label)

#training the svm model with face embeddings and labels 
classi=SVC(kernel='linear',probability=True)
classi.fit(u_pic,u_label)

#saving the trained model using pickle
c_file='svm_model.sav'
pickle.dump(classi, open(c_file, 'wb'))


#Testing part
#loading the saved model
c_file='svm_model.sav'
loaded_m = pickle.load(open(c_file, 'rb'))

#recognizing faces in live video 
capt=cv.VideoCapture(0)
while 1:
    _,img=capt.read()
            
    h, w = img.shape[:2]
    blob = cv.dnn.blobFromImage(cv.resize(img, (250, 250)), 1.0,
    (250, 250), (104.0, 117.0, 123.0))
    net.setInput(blob)
    faces = net.forward()
    print(faces.shape[2])
    #to draw faces on image
    for i in range(faces.shape[2]):
            confidence = faces[0, 0, i, 2]
            print(confidence)
            print('---------------')
            if confidence > 0.5:
                box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                cv.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 2)
                gray_image=img[y:y1,x:x1]
                test_pic=cv.resize(gray_image,(160,160),cv.INTER_AREA)
                test_pic=test_pic.astype('float32')
                test_pic=test_pic/255.0
                mean=test_pic.mean()
                std=test_pic.std()
                test_pic=(test_pic-mean)/std
                test_pic=np.expand_dims(test_pic,axis=0)
                test_em=modell.predict(test_pic)                 
                result=loaded_m.predict(test_em)
                print(result)
                if result<=0.8:
                    cv.putText(img,'unknown',(x,y1),cv.FONT_HERSHEY_SIMPLEX,1, (0,255,40), 3)
                else:
                    if result[0]==0:
                        cv.putText(img,'face3',(x,y1),cv.FONT_HERSHEY_SIMPLEX,1, (0,255,40), 3)
                    elif result[0]==1:
                        cv.putText(img,'ajay',(x,y1),cv.FONT_HERSHEY_SIMPLEX,1, (0,255,40), 3)
                    elif result[0]==2:
                        cv.putText(img,'face2',(x,y1),cv.FONT_HERSHEY_SIMPLEX,1, (0,255,40), 3)
    
    cv.imshow('detected', img)
    k=cv.waitKey(10)
    if k== 27:
        break
capt.release()
cv.destroyAllWindows()

