#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D,Rescaling,MaxPooling2D,Flatten,Dense
import keras
from keras import regularizers

import HyperParameters as hp_cls

class ConvolutionalNN:
    def __init__(self,hp = hp_cls.Hyperparameters()):
        self.hp = hp
    
    def create_cnn_model(self,num_classes, input_shape = (256,256,3)):
        inputs = layers.Input(shape=input_shape)
        rescaled_input = layers.Resizing(self.hp.image_size, self.hp.image_size)(inputs)
        
        # Block 1
        x = Conv2D(16, (3, 3), activation='gelu', padding='same', name='block1_conv1')(rescaled_input)
        x = Conv2D(16, (3, 3), activation='gelu', padding='same', name='block1_conv2',kernel_regularizer=regularizers.L2(0.01))(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
        # x = layers.Dropout(0.1)(x)
    
        # Block 2
        x = Conv2D(32, (3, 3), activation='gelu', padding='same', name='block2_conv1')(x)
        x = Conv2D(32, (3, 3), activation='gelu', padding='same', name='block2_conv2',kernel_regularizer=regularizers.L2(0.01))(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
        # x = layers.Dropout(0.1)(x)
        
        # Block 3
        x = Conv2D(64, (3, 3), activation='gelu', padding='same', name='block3_conv1')(x)
        x = Conv2D(64, (3, 3), activation='gelu', padding='same', name='block3_conv2')(x)
        x = Conv2D(64, (3, 3), activation='gelu', padding='same', name='block3_conv3')(x)
        x = Conv2D(64, (3, 3), activation='gelu', padding='same', name='block3_conv4',kernel_regularizer=regularizers.L2(0.01))(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
        # x = layers.Dropout(0.1)(x)

        # Block 4
        x = Conv2D(128, (3, 3), activation='gelu', padding='same', name='block4_conv1')(x)
        x = Conv2D(128, (3, 3), activation='gelu', padding='same', name='block4_conv2')(x)
        x = Conv2D(128, (3, 3), activation='gelu', padding='same', name='block4_conv3')(x)
        x = Conv2D(128, (3, 3), activation='gelu', padding='same', name='block4_conv4',kernel_regularizer=regularizers.L2(0.01))(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
        # x = layers.Dropout(0.1)(x)

        # Block 5
        x = Conv2D(128, (3, 3), activation='gelu', padding='same', name='block5_conv1')(x)
        x = Conv2D(128, (3, 3), activation='gelu', padding='same', name='block5_conv2')(x)
        x = Conv2D(128, (3, 3), activation='gelu', padding='same', name='block5_conv3')(x)
        x = Conv2D(128, (3, 3), activation='gelu', padding='same', name='block5_conv4',kernel_regularizer=regularizers.L2(0.01))(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
        # x = layers.Dropout(0.1)(x)

        
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='gelu', name='fc1',kernel_regularizer=regularizers.L2(0.01))(x)
        # x = layers.Dropout(0.1)(x)
        x = Dense(4096, activation='relu', name='fc2',kernel_regularizer=regularizers.L2(0.01))(x)
        # x = layers.Dropout(0.1)(x)
        x = Dense(num_classes, activation='softmax', name='predictions')(x)
        
        # Create the Keras model.
        model = keras.Model(inputs=inputs, outputs=x)
        return model