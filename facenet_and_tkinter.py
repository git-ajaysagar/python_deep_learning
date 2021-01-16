#multiple face recognition on live video using facenet and opencv dnn with tkinter gui
#the program is already trained on svm for three different faces

# -*- coding: utf-8 -*-
'''By Ajay'''


from tensorflow import keras
import cv2 as cv
import os
import numpy as np
import random
import pickle
from PIL import ImageTk
from tkinter import *
from PIL import Image as pim
from facenet_model import *

modell= keras.models.load_model(r'C:\Users\Ajay\Downloads\facenet_keras.h5')

modelFile = "C:\\Users\\Ajay\\models\\res10_300x300_ssd_iter_140000.caffemodel"
configFile = "C:\\Users\\Ajay\\models\\deploy.prototxt"
net = cv.dnn.readNetFromCaffe(configFile, modelFile)

c_file='svm_model.sav'
loaded_m = pickle.load(open(c_file, 'rb'))

k=Tk()
k.config(background='pink')

new_face=[]        

def training_model():
    lab9=Label(k3,text='Training the Model!')
    lab9.place(relx=0.15,rely=0.6)

    folder_name=new_face[0]
    c_file=training(folder_name)
    print(c_file)
    
heading=Label(k,font=('Arial',15,'bold'),text='Face Recognizer',bg='#FC773E')
heading.place(relx=0.45, rely=0.13)

lab1=Label(k,text='Add an unknown person?',font=('arial','15','bold'))
lab1.place(relx=0.75,rely=0.55)
lab2=Label(k,text='Name: ')
lab2.place(relx=0.75,rely=0.63)
ent1=Entry(k,width=20)
ent1.place(relx=0.79,rely=0.63)

new_var=1
def face():
    global gray_image
    _,img2=cc2.read()
    h, w = img2.shape[:2]
    blob = cv.dnn.blobFromImage(cv.resize(img2, (300, 300)), 1.0,
    (300, 300), (104.0, 117.0, 123.0))
    net.setInput(blob)
    faces = net.forward()
    #to draw faces on image
    for i in range(faces.shape[2]):
        try:
            confidence = faces[0, 0, i, 2]
            if confidence > 0.5:
                box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                cv.rectangle(img2, (x, y), (x1, y1), (0, 0, 255), 2)
                gray_image=img2[y:y1,x:x1]
                gray_image=cv.resize(gray_image,(160,160),cv.INTER_AREA)
    
        except:
            pass
    lab8=Button(k3,text='Train!',bg='gray',width=15,command=training_model)
    lab8.place(relx=0.15,rely=0.7)
    lab7=Label(k3,text='Click few pictures!',font=('arial','25','bold'))
    lab7.place(relx=0.4,rely=0.1)
    lab6=Button(k3,text='Click!',width=15,bg='gray',command=save_face)
    lab6.place(relx=0.75,rely=0.7)
    lab5=Button(k3,text='Close X',command=close_k3,bg='#fc3e64')
    lab5.place(relx=0.5,rely=0.9)
    img2=cv.cvtColor(img2,cv.COLOR_BGR2RGBA)
    im2=pim.fromarray(img2) 
    im2=ImageTk.PhotoImage(image=im2)
    ii2.image=im2
    ii2.configure(image=im2)
    ii2.after(100,face)
    k3.mainloop()

def save_face():
    alpha=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    r1=random.choice(alpha)
    r2=random.choice(alpha)
    r3=random.choice(alpha)
    r4=random.choice(alpha)
    pathh=new_face[0]
    print(pathh)
    new_path=os.path.join('F:\\Tkinter testing',pathh)
    os.chdir(new_path)
    cv.imwrite(r1+r2+r3+r4+'.jpg',gray_image)
def but_two():
    k.destroy()
    capt.release() 
    global k3
    k3=Tk()
    k3.config(background='pink')
    imageFrame2 = Frame(k3)
    imageFrame2.place(relx=0.5,rely=0.18,anchor='n')
    global ii2
    ii2 = Label(imageFrame2)
    ii2.grid(row=0, column=0)
    global cc2
    cc2=cv.VideoCapture(0)
    face()

def close_k3():
    k3.destroy()
    cc2.release()

def but_one():
    
    if ent1.get()=="":
            cc=Label(k,text='No name mentioned!')
            cc.place(relx=0.538,rely=0.56)
    else:
        c=Label(k,text=ent1.get()+' will be added!           ')
        c.place(relx=0.77,rely=0.72)
        partial_path='F:\Tkinter testing'
        folder=ent1.get()
        new_face.append(folder)
        path=os.path.join(partial_path,folder)
        os.mkdir(path)
        c.after(1500,but_two)
but1=Button(k,text='Submit',command=but_one)
but1.place(relx=0.79,rely=0.67)
    
imageFrame = Frame(k)
imageFrame.place(relx=0.5,rely=0.18,anchor='n')

ii = Label(imageFrame)
ii.grid(row=0, column=0)

capt=cv.VideoCapture(0)
def face_rec():
    _,img=capt.read()
    aj=[]
    aj.clear()
    ch=[]
    ch.clear()
    jy=[]
    jy.clear()
    listo=[]
    listo.clear()
    h, w = img.shape[:2]
    blob = cv.dnn.blobFromImage(cv.resize(img, (250, 250)), 1.0,
    (250, 250), (104.0, 117.0, 123.0))
    net.setInput(blob)
    faces = net.forward()
    print(faces.shape[2])
    #to draw faces on image
    for i in range(faces.shape[2]):
        try:
            confidence = faces[0, 0, i, 2]
            if confidence > 0.5:
                listo.append('face')
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
                if result<=0.8:
                    cv.putText(img,'unknown',(x,y1),cv.FONT_HERSHEY_SIMPLEX,1, (0,255,40), 3)                    
                else:
                    if result[0]==0:
                        aj.append('face3')
                        cv.putText(img,'face3',(x,y1),cv.FONT_HERSHEY_SIMPLEX,1, (0,255,40), 3)
                    elif result[0]==1:
                        aj.append('ajay')
                        cv.putText(img,'ajay',(x,y1),cv.FONT_HERSHEY_SIMPLEX,1, (0,255,40), 3)
                    elif result[0]==2:
                        aj.append('face2')
                        cv.putText(img,'face2',(x,y1),cv.FONT_HERSHEY_SIMPLEX,1, (0,255,40), 3)
        except:
            pass
    l2=Label(k)
    l2.place(relx=0.75,rely=0.3)
    if len(listo)==0:
        l1=Label(k,text=' No face detected!  ',font=('arial' ,20),bg='#d279a6')
        l1.place(relx=0.75,rely=0.2)
    else:
        l1=Label(k,text='Face detected o_O ',font=('arial',20),bg='#d279a6')
        l1.place(relx=0.75,rely=0.2)
    if len(aj)==0:
        l2.config(text=('   hello....                      '),font=('arial' ,15),bg='#d279a6')
    else:
        l2.config(text=('hello',aj),font=('arial' ,15),bg='#d279a6')
    img=cv.cvtColor(img,cv.COLOR_BGR2RGBA)
    im=pim.fromarray(img) 
    im=ImageTk.PhotoImage(image=im)
    ii.image=im
    ii.configure(image=im)
    ii.after(10,face_rec)
        
def close():
    k.destroy()
    
en=Button(k,text='close x',command=close,bg='#fc3e64')
en.place(relx=0.5,rely=0.9)

face_rec()
k.mainloop()
capt.release()

