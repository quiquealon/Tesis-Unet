#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import sys
import random
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images, imsave
from skimage.transform import resize
from skimage.morphology import label


from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import tensorflow as tf


# In[8]:


# Altura y Ancho de la imagen y los canales RGB
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

TRAIN_PATH = '/home/quiquealon/Documents/TIA/UNET/ISIC2018_Task1-2_Training_Input/'
MASK_PATH = '/home/quiquealon/Documents/TIA/UNET/ISIC2018_Task1_Training_GroundTruth/'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed


# In[3]:


# Tomar los datos para train y test

train_ids = next(os.walk(TRAIN_PATH))[2]


X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)


# In[14]:


print('Obteniendo y redimensionando imágenes de entrenamiento y máscaras. ... ')

sys.stdout.flush()

for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    
    path = TRAIN_PATH + id_
    img = imread(path)[:,:,:IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_train[n] = img
    
    #Redimensionar la mascara la mascara
    
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    path_mask=MASK_PATH + id_[:-4] + '_segmentation.png'
    img_mask = imread(path_mask)
    img_mask = np.expand_dims(resize(img_mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant',preserve_range=True), axis=-1)
    
    Y_train[n]=img_mask
    
    

print('Preprocesamiento Terminado!')


# In[ ]:





# In[10]:


# Guardar la data redimensionada del train

Y_train_int=np.array(np.squeeze(Y_train),dtype='int')

for id_ in tqdm(range(len(X_train)),total=len(X_train)):
    imsave('ISIC2018_Task1-2_Training_Input - redimensionada/' + train_ids[id_],X_train[id_])
    imsave('ISIC2018_Task1_Training_GroundTruth- redimensionada/' + train_ids[id_][:-4]+ '_segmentation.png',Y_train_int[id_])


# In[10]:


# Altura y Ancho de la imagen y los canales RGB para las imagenes ya redimensionadas
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
TRAIN_PATH = '/home/quiquealon/Documents/TIA/UNET/ISIC2018_Task1-2_Training_Input-redimensionada/'
MASK_PATH = '/home/quiquealon/Documents/TIA/UNET/ISIC2018_Task1_Training_GroundTruth-redimensionada/'


warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed


# In[5]:


train_ids = next(os.walk(TRAIN_PATH))[2]


X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)


# In[6]:


train_ids


# In[11]:


print('Obteniendo las imágenes ya redimensionadas de entrenamiento y máscaras. ... ')

sys.stdout.flush()

for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    
    path = TRAIN_PATH + id_
    img = imread(path)[:,:,:IMG_CHANNELS]
    X_train[n] = img
    
      
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    path_mask=MASK_PATH + id_[:-4] + '_segmentation.png'
    img_mask = imread(path_mask)
    img_mask = np.expand_dims(img_mask, axis=-1)
    mask = np.maximum(mask, img_mask)
    
    Y_train[n]=img_mask
    
    

print('Terminado!')


# In[13]:


# Ver la data para verificar que esta leyendo bien

ix = random.randint(0, len(train_ids))
imshow(X_train[ix])
plt.show()
imshow(np.squeeze(Y_train[ix]))
plt.show()


# In[41]:


# Metrica Jaccard o IoU (Intersection sobre union) 

def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


# In[7]:


# Metrica Dice
smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


# In[8]:


# Funcion de perdida Dice
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


# In[9]:


# Modelo de la U-Net

inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = Lambda(lambda x: x / 255) (inputs)

c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
c1 = Dropout(0.1) (c1)
c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
c2 = Dropout(0.1) (c2)
c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
c3 = Dropout(0.2) (c3)
c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
c4 = Dropout(0.2) (c4)
c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
c5 = Dropout(0.3) (c5)
c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
c6 = Dropout(0.2) (c6)
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
c7 = Dropout(0.2) (c7)
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
c8 = Dropout(0.1) (c8)
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
c9 = Dropout(0.1) (c9)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef])
model.summary()


# In[49]:


# Fit modelo
# earlystopper = EarlyStopping(patience=5, verbose=1)
# checkpointer = ModelCheckpoint('Modelo_v001.h5', verbose=1, save_best_only=True)
checkpointer = ModelCheckpoint('Modelo_v001.h5', verbose=1, save_best_only=True)
results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=50,callbacks=[checkpointer])

model.save('Modelo_v001_final.h5')


# In[51]:


# Path de las imagenes para el test

TEST_PATH = '/home/quiquealon/Documents/TIA/UNET/ISIC2018_Task1-2_Validation_Input/'
test_ids = next(os.walk(TEST_PATH))[2]


# In[53]:


# Redimensionar Imagenes para el test

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print('Obteniendo y redimensionando imágenes de prueba. ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    img = imread(path)[:,:,:IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img
    
print('Preprocesamiento Terminado!')


# In[55]:


# Guardar la data redimensionada del test

for id_ in tqdm(range(len(X_test)),total=len(X_test)):
    imsave('ISIC2018_Task1-2_Validation_Input-redimensionada/' + test_ids[id_],X_test[id_])
   


# In[10]:


# Obtener imagenes redimensionadas para el test

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

TEST_PATH = '/home/quiquealon/Documents/TIA/UNET/ISIC2018_Task1-2_Validation_Input-redimensionada/'
test_ids = next(os.walk(TEST_PATH))[2]


X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print('Obteniendo imágenes redimensionadas de prueba. ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    img = imread(path)[:,:,:IMG_CHANNELS]
    X_test[n] = img
    
print('Carga Terminada!')


# In[11]:


# Cargar modelo

model = load_model('Modelo_v001_final.h5', custom_objects={'dice_coef': dice_coef})


# In[12]:


# Predecir en validation test

preds_test = model.predict(X_test, verbose=1)


# In[13]:


preds_test_t = (preds_test > 0.5).astype(np.uint8)


# In[19]:


ix = random.randint(0, len(X_test))

imshow(X_test[ix])
plt.show()

imshow(np.squeeze(preds_test[ix]))
plt.show()

imshow(np.squeeze(preds_test_t[ix]))
plt.show()


# In[15]:


# Predecir en el train

preds_train = model.predict(X_train, verbose=1)


# In[16]:


preds_train_t = (preds_train > 0.5).astype(np.uint8)


# In[17]:


ix = random.randint(0, len(preds_train_t))

imshow(X_train[ix])
plt.show()

imshow(np.squeeze(Y_train[ix]))
plt.show()

imshow(np.squeeze(preds_train_t[ix]))
plt.show()

