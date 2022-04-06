import encodings
from turtle import width
import cv2 #Top 1 librerias de imagenes
import pandas as pd
import numpy as np
import os


from keras.utils import np_utils 
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras import layers
from keras.layers import Dropout
from sklearn.model_selection import train_test_split


dir_1='Datasets/'
nom_b='/'
ext_b2=['.jpeg',".jpg",".png",".bmp",".JPG"]
ext_b=".jpeg"
contenido = os.listdir(dir_1)
print(contenido)
carp_v=[]

#Vector de las carpetas

for cont in contenido:
  spl = cont.split(".")
  if len(spl)==1:
    carp_v.append('/'+cont)

#Creacion de datasets


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




#Sumarlos al vector 
c_img=[]
img_v=[]
label=[]
count=0

for con,serie in zip(carp_v,range(len(carp_v))):
  if  os.path.exists(dir_1+"/"+str(serie)):
    c_img.append(str(serie))



for m,n in zip(c_img,range(len(c_img))):
  imag_c = os.listdir(dir_1+'/'+m)
  count+=1
  for im_c,num_im in zip(imag_c,range(len(imag_c))):
    spl = im_c.split(".")
    for ext_i in ext_b2:
      if ext_i=="."+spl[-1]:
        im = cv2.imread(dir_1+'/'+m+"/"+im_c)
        img_v.append(im)
        label.append(m)
      else:
        pass

print(len(img_v))
print(len(label))

#Red neuronal



def plot(h):
      LOSS = 0; ACCURACY = 1
      training = np.zeros((2,epochs_v))
      testing = np.zeros((2,epochs_v))
      training[LOSS] = h.history['loss']
      testing[LOSS] = h.history['val_loss']    # validation loss
      training[ACCURACY] = h.history['accuracy']
      testing[ACCURACY] = h.history['val_accuracy']  # validation accuracy

      epochs = range(1,epochs_v+1)
      fig, axs = plt.subplots(1,2, figsize=(17,5))
      for i, label in zip((LOSS, ACCURACY),('loss', 'accuracy')):   
          axs[i].plot(epochs, training[i], 'b-', label='Training ' + label)
          axs[i].plot(epochs, testing[i], 'y-', label='Test ' + label)
          axs[i].set_title('Training and test ' + label)
          axs[i].set_xlabel('Epochs')
          axs[i].set_ylabel(label)
          axs[i].legend()
      plt.show()
      print(np.mean(training[ACCURACY]))
      print(np.mean(testing[ACCURACY]))


X_train, X_test, y_train, y_test = train_test_split( img_v, label, test_size=0.2, random_state=42)

X_train=np.asarray(X_train)
print(len(X_train))
print(len(X_train[0]))
print(len(X_train[0][0]))
print(len(X_train[0][0][0]))

y_train=np.asarray(y_train)

X_test=np.asarray(X_test)

y_test=np.asarray(y_test)


num_train_samples = X_train.shape[0]
num_test_samples = X_test.shape[0]
image_height = X_train.shape[1]
image_width = X_train.shape[2]
num_channels = X_train.shape[3]
#print(num_train_samples, image_height, image_width, num_channels)
X_train = X_train / 255
X_test = X_test / 255

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
#print(len(y_train[0]))
num_clases = len(y_train[0])

model = Sequential()
model.add(layers.Conv2D(16, (3,3), activation='relu', input_shape=(image_height,image_width, num_channels)))
model.add(layers.MaxPool2D(2,2))
model.add(layers.Conv2D(32, (3,3), activation='relu'))
#model.add(layers.MaxPool2D(2,2))
#model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(layers.Dense(64))
model.add(layers.Dense(48))
model.add(layers.Dense(num_clases,activation='softmax'))
print(model.summary())
count=0
for i in [500]: #Epocas
    for j in [512]: #Batch
      count+=1
      epochs_v = i
      batch=j
      model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])
      history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs_v, batch_size=batch,verbose=0)
      print('Numero Batch: ', batch)
      print('Numero epoc: ', epochs_v)
      print(count)
      plot(history)

model_json=model.to_json()
with open ('model.json','w') as json_file:
  json_file.write(model_json)
model.save_weights('par.h5')

print('guardado')

#Agradecimientos para BIT y aquamary13
