# Tesis-Unet

# Trabajo para el curso de Seminario de Tesis

## Objetivos



1. El objetivo general planteado es utilizar una U-Net multiescala para la segmentación semántica de imágenes dermatoscópicas para el dataset del ISIC 2018 para la tarea 1.

2. Localizar el área de la lesión en la imagen con el objetivo de recortarla.

3. Redimensionar las imágenes a un tamaño de 512x512 píxeles.

4. Dividir la imagen en 4 parches diferentes cada uno de tamaño de 256x256 píxeles.

5. Realizar las operaciones previas a la U-net (redimensionamiento utilizando UpSampling2D  y MaxPool2D)
