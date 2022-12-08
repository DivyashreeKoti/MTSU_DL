#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import Patches as patch_cls

from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot

#Funtion to display the images in Dataset
def plot_patch(datagen,hp,model,version):   
    
    # load the image
    img = load_img('./Data/traffic_Data/TRAIN/1/001_0002.png')
    # convert to numpy array
    data = img_to_array(img)
    # expand dimension to one sample
    samples = expand_dims(data, 0)
    # create image data augmentation generator
    datagen = ImageDataGenerator(rotation_range=65,
                                  width_shift_range=0.4,
                                  height_shift_range=0.1,
                                  brightness_range=[0.3,0.9],
                                  zoom_range=0.25,
                                  vertical_flip=True,
                                  horizontal_flip = True)
    # prepare iterator
    it = datagen.flow(samples, batch_size=1)
    # generate samples and plot
    for i in range(9):
        # define subplot
        pyplot.subplot(330 + 1 + i)
        # generate batch of images
        batch = it.next()
        # convert to unsigned integers for viewing
        image = batch[0].astype('uint8')
        # plot raw pixel data
        pyplot.imshow(image)
    # show the figure
    pyplot.show()
    pyplot.savefig('./image/'+model+'_augmented.png')

    #Display patches of images
    plt.figure(figsize=(4, 4))
    
    image = mpimg.imread('./Data/traffic_Data/TRAIN/1//001_0002.png')
    plt.imshow(image)
    plt.axis("off")
    plt.savefig('./image/'+model+'_'+version+'_image.png')
    resized_image = tf.image.resize(
        tf.convert_to_tensor([image]), size=(hp.image_size, hp.image_size)
    )
    
    
    patches = patch_cls.Patches(hp.patch_size)(resized_image)

    n = int(np.sqrt(patches.shape[1]))
    plt.figure(figsize=(4, 4))
    for i, patch in enumerate(patches[0]):
        ax = plt.subplot(n, n, i + 1)
        patch_img = tf.reshape(patch, (hp.patch_size, hp.patch_size, 3))
        plt.imshow(patch_img.numpy())
        plt.axis("off")
    plt.savefig('./image/'+model+'_'+version+'_patches.png')
    
    #Display augmented image
    plt.figure()
    fig, axs = plt.subplots(3, 2)
    img = mpimg.imread('./Data/traffic_Data/TRAIN/36/036_0001.png')
    axs[0,0].imshow(img)
    img = mpimg.imread('./Data/traffic_Data/TRAIN/30/030_0001.png')
    axs[0,1].imshow(img)
    img = mpimg.imread('./Data/traffic_Data/TRAIN/3/003_0001.png')
    axs[1,0].imshow(img)
    img = mpimg.imread('./Data/traffic_Data/TRAIN/18/018_0001.png')
    axs[1,1].imshow(img)
    img = mpimg.imread('./Data/traffic_Data/TRAIN/4/004_0001.png')
    axs[2,0].imshow(img)
    img = mpimg.imread('./Data/traffic_Data/TRAIN/19/019_0001.png')
    axs[2,1].imshow(img)

    plt.show()
    plt.savefig('./image/Main.png')
    
    
