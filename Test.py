#Import the libreries
import json
import cv2 
import pandas as pd
import numpy as np
import os
from keras.utils import np_utils 
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras import layers
from keras.layers import Dropout
from sklearn.model_selection import train_test_split
import tensorflow as tf


#Find the test images for predict 
dir_1='Test/'
nom_b='/'
ext_b2=['.jpeg',".jpg",".png",".bmp",".JPG"]
ext_b=".jpeg"
contenido = os.listdir(dir_1)
print(contenido)
carp_v=[]

#Vector of the images

for cont in contenido:
  spl = cont.split(".")
  if len(spl)==1:
    carp_v.append('/'+cont)

#Reduce the images size


for carp,cont_c in zip(carp_v,range(len(carp_v))):
     imag_c = os.listdir(dir_1+carp)
     print(carp,imag_c)
     if os.path.exists(dir_1+"/"+str(cont_c)):
        print("la carpeta existe")
     else:
        os.mkdir(dir_1+"/"+str(cont_c))
     #################   
     for im_c,num_im in zip(imag_c,range(len(imag_c))):
       spl = im_c.split(".")
       for ext_i in ext_b2:
         if ext_i=="."+spl[-1]:   
           im = cv2.imread(dir_1+carp+"/"+im_c)
           #try:
             #im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
           #except:
             #print(" es gris") 
           #im =  np.array(im)  
           try:
              im = cv2.resize(im,(28,28), interpolation=cv2.INTER_CUBIC)
           except:
              im=im 
           try:      
              cv2.imwrite(dir_1+"/"+str(cont_c)+"/"+str(num_im)+ext_b,im)
           except:
              print(" no guardo")   



#Add it to the vector
c_ima=[]
img_vec=[]
label=[]
count=0

for con,serie in zip(carp_v,range(len(carp_v))):
  if  os.path.exists(dir_1+"/"+str(serie)):
    c_ima.append(str(serie))


for m,n in zip(c_ima,range(len(c_ima))):
  imag_c = os.listdir(dir_1+'/'+m)
  count+=1
  for im_c,num_im in zip(imag_c,range(len(imag_c))):
    spl = im_c.split(".")
    for ext_i in ext_b2:
      if ext_i=="."+spl[-1]:
        im = cv2.imread(dir_1+'/'+m+"/"+im_c)
        img_vec.append(np.array(im))
        label.append(m)
      else:
        pass

print(len(img_vec))


#Import the models and parameters
dir_F='par.h5'

json_file=open('model.json','r')
loaded_model_json=json_file.read()
json_file.close()
loaded_model=tf.keras.models.model_from_json(loaded_model_json)


loaded_model.load_weights(dir_F)
print(loaded_model.summary())
predict=loaded_model.predict(np.array(img_vec))


#Create the label and modify 
a=np.ones(5)*2
b=np.ones(5)*3
c=np.ones(5)*4
d=np.ones(5)*5

etiqueta=np.concatenate((a,b,c,d))
etiqueta=etiqueta.tolist()
etiqueta=[int(etiqueta) for etiqueta in etiqueta]
print(etiqueta)

print(predict)

#The prediction acurracy
def buscar(label,predict):
  count=0
  for i in range(len(label)):
    if predict[i][label[i]]:

      count+=1
    else:
      pass
  print('La precisión fue de:',str(count/len(label)*100),'%')

buscar(etiqueta,predict)





