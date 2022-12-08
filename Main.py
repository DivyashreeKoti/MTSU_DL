#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


import tensorflow.keras as keras


import LoadData as ld
import Patches as patch
import HyperParameters as hp_cls
import ExperimentSetup as es
import ConvolutionalNN as cnn
import ViTransformer as vit_main
import Realformer as realformer
import VGG19 as vgg19_cls

#Start of Main
if __name__ == "__main__":
    strategy = tf.distribute.OneDeviceStrategy('gpu:1')
    
    #setting to run on GPU
    with strategy.scope():
        #Loading data using Datagenerator
        ds_obj = ld.DataLoad()
        train_gen, validation_gen, test_gen = ds_obj.load_data(data_type='no augment')
        train_gen_aug, validation_gen_aug,test_gen_aug = ds_obj.load_data(data_type='augment',
                                                          rotation_range=65,
                                                          width_shift_range=0.4,
                                                          height_shift_range=0.1,
                                                          brightness_range=[0.3,0.7],
                                                          zoom_range=0.25,
                                                          horizontal_flip=True,
                                                          vertical_flip=True)


        
        # check iterators are working
        batchX, batchY = train_gen.next()
        batchX_aug, batchY_aug = train_gen_aug.next()
        
        input_shape = batchX.shape[1:]

        
        

        print('List of arguments:', (sys.argv))
        
        num_classes = len(ds_obj.classes) #getting number of classes
        
        #to opt to train or test and which model or all models
        if(sys.argv[1].lower() == 'train'):
            if(sys.argv[2].lower() == 'vit' or sys.argv[2].lower() == 'all'):
                vit_main.ViTransformer(num_classes,train_gen,validation_gen,train_gen_aug,validation_gen_aug)
            if(sys.argv[2].lower() == 'realformer' or sys.argv[2].lower() == 'all'):
                realformer.Realformer(num_classes,train_gen,validation_gen,train_gen_aug,validation_gen_aug)
            if(sys.argv[2].lower() == 'vgg' or sys.argv[2].lower() == 'all'):
                vgg19_cls.VGG19(num_classes,train_gen,validation_gen,train_gen_aug,validation_gen_aug)
        else:
            if(sys.argv[2].lower() == 'vit' or sys.argv[2].lower() == 'all'):
                print('\n\n************ ViT START ************')
                vit_main.ViTransformer_Test(num_classes, test_gen, test_gen_aug)
                print('\n\n************ ViT END ************')
            if(sys.argv[2].lower() == 'realformer' or sys.argv[2].lower() == 'all'):
                print('\n\n************ Realformer START ************')
                realformer.Realformer_Test(num_classes, test_gen, test_gen_aug)
                print('\n\n************ Realformer END ************')
            if(sys.argv[2].lower() == 'vgg' or sys.argv[2].lower() == 'all'):
                print('\n\n************ VGG START ************')
                vgg19_cls.VGG19_Test(num_classes, test_gen, test_gen_aug)
                print('\n\n************ VGG END ************')

    