#!/usr/bin/env python3
import os
import keras
import numpy as np
import tensorflow_addons as tfa
import tensorflow as tf

import HyperParameters as hp_cls

#Funtion to fit the training set
def run_experiment(model,train_generator,validation_generator, model_name,hp = hp_cls.Hyperparameters()):
    if vgg not in model_name.lower():
        optimizer = tfa.optimizers.AdamW(
            learning_rate=hp.learning_rate, weight_decay=hp.weight_decay
        )
    else:
        optimizer = tf.optimizers.Adam(
            lr=hp.learning_rate, decay=hp.weight_decay
        )
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    # print(model.summary())
    checkpoint_filepath = "/tmp/checkpoint"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )
    batch_size = hp.batch_size
    train_images = train_generator.samples
    test_images = validation_generator.samples
    
    checkpoint_path = "training_2/"+model_name+".ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,save_freq=10*batch_size,
                                                         verbose=2)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
    history = model.fit(
        train_generator,
        steps_per_epoch=train_images/batch_size,
        epochs=hp.num_epochs,
        # shuffle = True,
        validation_data=validation_generator,
        validation_steps=test_images/batch_size,verbose=1,
        callbacks=[early_stop,cp_callback]
    )

    return history

def test_model(model, test_generator,model_name,hp):
    optimizer = tfa.optimizers.AdamW(
        learning_rate=hp.learning_rate, weight_decay=hp.weight_decay
    )
    checkpoint_path = "training_2/"+model_name+".ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    print(os.listdir(checkpoint_dir))
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    # # Evaluate the model

    # Loads the weights
    model.load_weights(checkpoint_path)

    # Re-evaluate the model
    loss, acc = model.evaluate(test_generator, verbose=1)
    print(model_name,"************ accuracy: {:5.2f}%".format(100 * acc))
    print(model_name,"************ loss: {:5.2f}%".format(loss))
    
    raw_predictions = model.predict(test_generator)
    predictions = []
    for item in raw_predictions:
        predictions.append(np.argmax(item))
        
    print("************ predicted: ", predictions)