#To be used with 'facenet_and_tkinter.py' file for adding new faces to the model
# -*- coding: utf-8 -*-
'''By Ajay'''

def training(folder_name):
    alphabets={' ':0,'a':1,'b':2,'c':3,'d':4,'e':5,'f':6,'g':7,'h':8,'i':9,'j':10,'k':11,'l':12,
               'm':13,'n':14,'o':15,'p':16,'q':17,'r':18,'s':19,'t':20,'u':21,'v':22,
               'w':23,'x':24,'y':25,'z':26}
    from tensorflow import keras
    import numpy as np
    import os
    import cv2 as cv
    import random
    import pickle
    from sklearn.svm import SVC
    from sklearn.preprocessing import Normalizer
    modell= keras.models.load_model(r'C:\Users\Ajay\Downloads\facenet_keras.h5')
    folder_path=os.path.join('F:\Tkinter testing',folder_name)
    folder_list=os.listdir(folder_path)
    print(folder_list)
    folder_pics=[]
    
    for f1 in folder_list:
        pic=cv.imread(os.path.join(folder_path,f1))
        print(pic.shape)
        np.array(pic)
        pic=pic.astype('float32')
        pic=pic/255.0
        mean=pic.mean()
        std=pic.std()
        pic=(pic-mean)/std
        pic=np.expand_dims(pic,axis=0)
        embed=modell.predict(pic) 
        folder_pics.append(embed[0])   
        
    folder_pics=np.array(folder_pics)
    print(folder_pics.shape)
    
    nor=Normalizer(norm='l2')
    folder_pics=nor.transform(folder_pics)
    alpha_nums=[]
    for n in folder_name:
        alpha_num=alphabets[n]
        alpha_nums.append(alpha_num)
    alpha_nums=np.array(alpha_nums)
    label=alpha_nums.sum()
    
    new_train=[]
    for u in folder_pics:
        new_train.append([u,label])                
        
    # print(np.array(new_train).shape)
    random.shuffle(new_train)
    u_pic=[]
    u_label=[]
    for i1,i2 in new_train:
        u_pic.append(i1)
        u_label.append(i2)
    
    u_pic=np.array(u_pic)
    u_label=np.array(u_label)
    
    classi=SVC(kernel='linear',probability=True)
    classi.fit(u_pic,u_label)
    c_file='facenet_tkinter_trained_model.sav'
    pickle.dump(classi, open(c_file, 'wb'))
    return c_file

