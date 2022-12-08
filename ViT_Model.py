#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.keras import layers
import keras
from keras import regularizers

import Patches as patch
import PatchEncoder as pe
import HyperParameters as hp_cls

input_shape = (256, 256, 3)

#Most part is adopted from keras. Paper includes the citaiton
def mlp(x, hidden_units, dropout_rate,regularization):
    for units in hidden_units:
        if(regularization == 'L2'):
            x = layers.Dense(units, activation=tf.nn.gelu,kernel_regularizer=regularizers.L2(0.01))(x)
        else:
            x = layers.Dense(units, activation=tf.nn.gelu)(x)
            x = layers.Dropout(dropout_rate)(x)
    return x

def create_vit_classifier(num_classes, model_variant = 'ViT',hp = hp_cls.Hyperparameters(),regularization = None):
    
    inputs = layers.Input(shape=input_shape)
    # Augment data.
    augmented = layers.Resizing(hp.image_size, hp.image_size)(inputs)
    # Create patches.
    patches = patch.Patches(hp.patch_size)(augmented)
    # Encode patches.
    encoded_patches = pe.PatchEncoder(hp.num_patches, hp.projection_dim)(patches)
    residual_attention_scores = None
    # Create multiple layers of the Transformer block.
    for _ in range(hp.transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        if(regularization == 'L2'):
            attention_output = layers.MultiHeadAttention(
                num_heads=hp.num_heads, key_dim=hp.projection_dim,kernel_regularizer=regularizers.L2(0.01),
            bias_regularizer=regularizers.L2(0.01))(x1, x1)
        else:
            attention_output = layers.MultiHeadAttention(
            num_heads=hp.num_heads, key_dim=hp.projection_dim, dropout=0.1
        )(x1, x1)

        if('realformer' in model_variant.lower()):
            # Add residual attention scores.
            if(residual_attention_scores == None):
                residual_attention_scores = attention_output
            else:
                residual_attention_scores = layers.Add()([attention_output, residual_attention_scores])

                # Add back scores to multi-head attention layer.
                attention_output = layers.Add()([attention_output, residual_attention_scores])
        
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=hp.transformer_units, dropout_rate=0.1,regularization=regularization)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    
    # Add MLP.
    features = mlp(representation, hidden_units=hp.mlp_head_units, dropout_rate=0.5,regularization=regularization)
    # Classify outputs.
    logits = layers.Dense(num_classes, activation='softmax')(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model