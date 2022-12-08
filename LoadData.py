#!/usr/bin/env python3
import pandas as pd
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class DataLoad:
    def __init__(self):
        label_path = './Data/labels.csv'
        self.classes = pd.read_csv(label_path)
        self.dataset_path = './Data/traffic_Data/TRAIN'
        self.testset_path = './Data/traffic_Data/TEST'
    
    def load_data(self,data_type,samplewise_std_normalization=False,
                  zca_whitening=False,
                  zca_epsilon=1e-06,
                  rotation_range=0,
                  width_shift_range=0.0,
                  height_shift_range=0.0,
                  brightness_range=None,
                  shear_range=0.0,
                  zoom_range=0.0,
                  channel_shift_range=0.0,
                  horizontal_flip=False,
                  vertical_flip=False,
                  rescale=1.0/255,
                  validation_split=0.75):
        #create imagedata generator for train and validation set
        datagen = ImageDataGenerator(samplewise_std_normalization=samplewise_std_normalization,
                  zca_whitening=zca_whitening,
                  zca_epsilon=zca_epsilon,
                  rotation_range=rotation_range,
                  width_shift_range=width_shift_range,
                  height_shift_range=height_shift_range,
                  brightness_range=brightness_range,
                  shear_range=shear_range,
                  zoom_range=zoom_range,
                  channel_shift_range=channel_shift_range,
                  horizontal_flip=horizontal_flip,
                  vertical_flip=vertical_flip,
                  rescale=rescale,
                  validation_split=validation_split)

        train_gen = datagen.flow_from_directory(self.dataset_path, class_mode='categorical',
                                                subset='training',shuffle=True)
        validation_gen = datagen.flow_from_directory(self.dataset_path, class_mode='categorical', subset='validation',
                                                     shuffle=False)
        
        #create imagedata generator for test set
        datagen2 = ImageDataGenerator(samplewise_std_normalization=samplewise_std_normalization,
                  zca_whitening=zca_whitening,
                  zca_epsilon=zca_epsilon,
                  rotation_range=rotation_range,
                  width_shift_range=width_shift_range,
                  height_shift_range=height_shift_range,
                  brightness_range=brightness_range,
                  shear_range=shear_range,
                  zoom_range=zoom_range,
                  channel_shift_range=channel_shift_range,
                  horizontal_flip=horizontal_flip,
                  vertical_flip=vertical_flip,
                  rescale=rescale,
                  validation_split=0.007)
        # test_gen = datagen2.flow_from_directory(self.testset_path, classes=['.'],class_mode='categorical',
        #                                              shuffle=False, batch_size=1)
        test_gen = datagen2.flow_from_directory(self.dataset_path, class_mode='categorical', subset='validation',
                                                     shuffle=False,seed=544)
        print('\n\n************ True Labels of',data_type,test_gen.classes)
        # for i in range(test_gen.samples):
        #     print('************ ',test_gen.filenames[i])
        # to see what images were selected
        fig, axis = plt.subplots(5,5)
        for i in range(5):
            for j in range(5):
                batch=next(test_gen)  # returns the next batch of images and labels 
                img=batch[0][i+j]
                axis[i,j].imshow(img)
        plt.savefig(data_type+'.png')
        return train_gen, validation_gen, test_gen