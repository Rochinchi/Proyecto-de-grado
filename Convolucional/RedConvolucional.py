#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import copy
from random import randint
from numpy import array
from numpy import argmax
import random

from sklearn.feature_selection import SelectFromModel
from tensorflow.keras import layers, models
import pickle
from keras import callbacks
from keras.callbacks import CSVLogger

#Función para randomizar cada clase
def mezclar(array_original):
    array = np.empty_like(array_original)
    n = len(array_original)
    alea = list(range(n))
    random.shuffle(alea)
    for i in range(n):
        array[i] = array_original[alea[i]]
    return array


# Separación de entrenamiento  y validación

def separar (array):
    val = 0.7 #Porcentaje de entrenamiento
    indx_test = round(array.shape[0]*val)
    array_test = array[0:indx_test,:]
    array_val = array[(indx_test):len(array),:]
    return array_test, array_val



#Onehot
def EncoderOneHot (Y):
    values = array(Y)
    label_encoder = preprocessing.LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    # binary encode
    onehot_encoder = preprocessing.OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    Y_o = onehot_encoder.fit_transform(integer_encoded)
    print(Y_o)
    return Y_o



version = "_MayCorr901_V2_C2" # Cambiar según la versión del código
nombre_archivo = f"Prueba{version}.h5"



# Extraer los valores de la base de datos

train = np.genfromtxt('malware_MCorTrainI90R1_V2.csv', delimiter=',')
val =np.genfromtxt('malware_MCorValI90R1_V2.csv', delimiter=',')



#maty = Data_cor.values[:, 0]
maty_train = train[:, 0]
maty_val   = val[:, 0]

maty_train = EncoderOneHot(maty_train)
maty_val = EncoderOneHot(maty_val)



#matx.astype(int)

matx_train = train[:, 1:]
matx_val = val[:, 1:]

 
# Genero la matriz de entrenamiento

Data_conv_ent = np.zeros((46877, 7, 8, 1))

Data_cor1_ent = np.concatenate((matx_train,np.zeros((46877,4))), axis=1)

for i in range(35158):
    Data_conv_ent[i,:,:,0] = np.array(Data_cor1_ent[i]).reshape(7, 8)

# Validacion
Data_conv_val = np.zeros((8789, 7, 8, 1))

Data_cor1_val = np.concatenate((matx_val,np.zeros((8789,4))), axis=1)

for i in range(8789):
    Data_conv_val[i,:,:,0] = np.array(Data_cor1_val[i]).reshape(7, 8)


# Convertir el DataFrame a un tensor de TensorFlow
tensor_ent = tf.convert_to_tensor(Data_conv_ent)
tensor_val = tf.convert_to_tensor(Data_conv_val)


""" --- Modelo convolucional --- """

model = models.Sequential([
    layers.Conv2D(64, (2, 2), activation='relu', input_shape=(7, 8, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (2, 2), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (1, 1), activation='relu'),
    layers.MaxPooling2D((1, 1)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(3, activation='softmax')
])


# Compilacion de modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


maty_train = maty_train.astype('float32')
maty_val = maty_val.astype('float32')
#type(maty)

yuca = f'Prueba{version}.log'
csv_logger = CSVLogger(yuca)
m1 = model.fit(tensor_ent, maty_train, epochs=2000, validation_data=(tensor_val, maty_val), callbacks=[csv_logger])


nombre_accuracy = f"Grafica{version}_Accuracy.png"
nombre_loss = f"Grafica{version}_Loss.png"


m1.model.save(nombre_archivo)
h4 = m1

plt.figure(0)
plt.ylabel('accuracy'); plt.xlabel('epoch')
plt.plot(h4.history['accuracy'],'r')
plt.plot(h4.history['val_accuracy'],'g')
plt.grid()
plt.legend(['Entrenamiento','Validación'])
plt.savefig(nombre_accuracy)

plt.figure(1)
plt.ylabel('loss'); plt.xlabel('epoch')
plt.plot(h4.history['loss'],'r')
plt.plot(h4.history['val_loss'],'g')
plt.grid()
plt.legend(['Entrenamiento','Validación'])
plt.savefig(nombre_loss)

plt.show



variables = {'matx': matx, 'maty': maty, 'Data_cor': Data_cor, 'Data_conv': Data_conv, 'm1': m1}

with open('Prueba{version}.pickle', 'wb') as f:
    pickle.dump(variables, f)








