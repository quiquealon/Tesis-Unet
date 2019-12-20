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
from skimage import measure
from skimage.util import crop


# In[181]:


TRAIN_PATH = '/home/quiquealon/Documents/TIA/UNET/ISIC2018_Task1-2_Training_Input/'
MASK_PATH = '/home/quiquealon/Documents/TIA/UNET/ISIC2018_Task1_Training_GroundTruth/'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed


# In[182]:


# Tomar los datos para train y test

train_ids = next(os.walk(TRAIN_PATH))[2]

X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)


# In[16]:


train_ids


# In[184]:


for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    #Imagen original
    path = TRAIN_PATH + id_
    img = imread(path)[:,:,:IMG_CHANNELS]
    #Mascara original               
    path_mask=MASK_PATH + id_[:-4] + '_segmentation.png'
    img_mask = imread(path_mask)
    #Encontrar los pixeles blancos de la mascara
    white_pixels = np.array(np.where(img_mask == 255))
    
    #Hacer crop tomando los X y Y extremos de los pixeles blancos y agregando 10 pixeles de padding
    img_cropped = img[min(white_pixels[0]):max(white_pixels[0]),min(white_pixels[1]):max(white_pixels[1])]
    mask_cropped = img_mask[min(white_pixels[0]):max(white_pixels[0]),min(white_pixels[1]):max(white_pixels[1])]
    
    #Guardar
    imsave('/home/quiquealon/Documents/TIA/UNET/scaling/ISIC2018_Task1-2_Training_Input/' + train_ids[n],img_cropped)
    imsave('/home/quiquealon/Documents/TIA/UNET/scaling/ISIC2018_Task1_Training_GroundTruth/' + train_ids[n][:-4]+ '_segmentation.png',mask_cropped)

    
    
   


# In[4]:


# Altura y Ancho de la imagen y los canales RGB
IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 3

TRAIN_PATH = '/home/quiquealon/Documents/TIA/UNET/scaling/ISIC2018_Task1-2_Training_Input/'
MASK_PATH = '/home/quiquealon/Documents/TIA/UNET/scaling/ISIC2018_Task1_Training_GroundTruth/'


# In[5]:


# Tomar los datos para train y test

train_ids = next(os.walk(TRAIN_PATH))[2]


X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)


# In[21]:


print('Obteniendo y redimensionando imágenes de entrenamiento y máscaras. ... ')

sys.stdout.flush()

for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    
    path = TRAIN_PATH + id_
    img = imread(path)[:,:,:IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
   
    
    #Redimensionar la mascara la mascara
    
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    path_mask=MASK_PATH + id_[:-4] + '_segmentation.png'
    img_mask = imread(path_mask)
    img_mask = resize(img_mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant',preserve_range=True)
    
    imsave('/home/quiquealon/Documents/TIA/UNET/scaling/ISIC2018_Task1-2_Training_Input-redimensionada/' + train_ids[n],img)
    imsave('/home/quiquealon/Documents/TIA/UNET/scaling/ISIC2018_Task1_Training_GroundTruth-redimensionada/' + train_ids[n][:-4]+ '_segmentation.png',img_mask)

    
    

print('Redimension Terminado!')


# In[11]:


TRAIN_PATH_SPLIT = '/home/quiquealon/Documents/TIA/UNET/scaling/ISIC2018_Task1-2_Training_Input-split/'
MASK_PATH_SPLIT = '/home/quiquealon/Documents/TIA/UNET/scaling/ISIC2018_Task1_Training_GroundTruth-split/'

for id_ in tqdm(train_ids,total=len(train_ids)):
    os.mkdir(TRAIN_PATH_SPLIT + id_[:-4])
    os.mkdir(MASK_PATH_SPLIT + id_[:-4])
        


# In[14]:


TRAIN_PATH_REDIMENSIONADA = '/home/quiquealon/Documents/TIA/UNET/scaling/ISIC2018_Task1-2_Training_Input-redimensionada/' 
MASK_PATH_REDIMENSIONADA = '/home/quiquealon/Documents/TIA/UNET/scaling/ISIC2018_Task1_Training_GroundTruth-redimensionada/'

for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    #Imagen original
    path = TRAIN_PATH_REDIMENSIONADA + id_
    img = imread(path)[:,:,:IMG_CHANNELS]
    #Mascara original               
    path_mask=MASK_PATH_REDIMENSIONADA + id_[:-4] + '_segmentation.png'
    img_mask = imread(path_mask)

    
          
    
    #Hacer crop img
    img_cropped_1 = img[0:256,0:256]
    img_cropped_2 = img[0:256,256:512]
    img_cropped_3 = img[256:512,0:256]
    img_cropped_4 = img[256:512,256:512]
    
    #Hacer crop mask
    mask_cropped_1 = img_mask[0:256,0:256]
    mask_cropped_2 = img_mask[0:256,256:512]
    mask_cropped_3 = img_mask[256:512,0:256]
    mask_cropped_4 = img_mask[256:512,256:512]
    
    
     
    
    # Guardar img
    
    imsave('/home/quiquealon/Documents/TIA/UNET/scaling/ISIC2018_Task1-2_Training_Input-split/'+train_ids[n][:-4]+'/'+train_ids[n][:-4]+'_1.jpg',img_cropped_1)
    imsave('/home/quiquealon/Documents/TIA/UNET/scaling/ISIC2018_Task1-2_Training_Input-split/'+train_ids[n][:-4]+'/'+train_ids[n][:-4]+'_2.jpg',img_cropped_2)  
    imsave('/home/quiquealon/Documents/TIA/UNET/scaling/ISIC2018_Task1-2_Training_Input-split/'+train_ids[n][:-4]+'/'+train_ids[n][:-4]+'_3.jpg',img_cropped_3)
    imsave('/home/quiquealon/Documents/TIA/UNET/scaling/ISIC2018_Task1-2_Training_Input-split/'+train_ids[n][:-4]+'/'+train_ids[n][:-4]+'_4.jpg',img_cropped_4)
    
    #Guardar mask

    imsave('/home/quiquealon/Documents/TIA/UNET/scaling/ISIC2018_Task1_Training_GroundTruth-split/'+train_ids[n][:-4]+'/'+train_ids[n][:-4]+'_1_segmentation.png',mask_cropped_1)
    imsave('/home/quiquealon/Documents/TIA/UNET/scaling/ISIC2018_Task1_Training_GroundTruth-split/'+train_ids[n][:-4]+'/'+train_ids[n][:-4]+'_2_segmentation.png',mask_cropped_2)
    imsave('/home/quiquealon/Documents/TIA/UNET/scaling/ISIC2018_Task1_Training_GroundTruth-split/'+train_ids[n][:-4]+'/'+train_ids[n][:-4]+'_3_segmentation.png',mask_cropped_3)
    imsave('/home/quiquealon/Documents/TIA/UNET/scaling/ISIC2018_Task1_Training_GroundTruth-split/'+train_ids[n][:-4]+'/'+train_ids[n][:-4]+'_4_segmentation.png',mask_cropped_4)
    

