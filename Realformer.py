#!/usr/bin/env python3
import tensorflow.keras as keras
import matplotlib.pyplot as plt

import HyperParameters as hp_cls
import ExperimentSetup as es
import ViT_Model as vit
import Plot_patch as pp

################# PLOT Accuracy and Loss #################
def plot_history(history,model='vit',version='base'):
    plt.figure()
    plt.plot(history.history['accuracy'], color = 'blue')
    plt.plot(history.history['val_accuracy'], color = 'green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.legend(["Accuracy","Validation Accuracy"])
    plt.savefig(model+'_'+version+'_accuracy_history.png')

    plt.figure()
    plt.plot(history.history['loss'], color = 'orange')
    plt.plot(history.history['val_loss'], color = 'red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend(["Loss","Validation Loss"])
    plt.savefig(model+'_'+version+'_loss_history.png')

################# TRAINING #################
def Realformer(num_classes,train_gen,validation_gen,train_gen_aug,validation_gen_aug):
    model='realformer'
    version='base'
    model_name = 'realformer'
    ################# BASE #################
    #fine tuning hyperparameters for Base model
    hp_base = hp_cls.Hyperparameters(image_size = 128,patch_size = 32)
    realformer_model = vit.create_vit_classifier(num_classes,hp=hp_base, model_variant = model_name)
    print(realformer_model.summary())
    # keras.utils.plot_model(realformer_model,show_shapes=True,expand_nested=True,to_file="realformer_model.png")
    realformer_history = es.run_experiment(realformer_model, train_gen, validation_gen,model_name)
    plot_history(realformer_history,model=model,version=version)
    
    ################# L2 REG #################
    model_name = 'realformer_l2'
    version='L2 reg'
    hp_base_L2 = hp_cls.Hyperparameters(image_size = 72,patch_size = 32,learning_rate = 0.0001)
    realformer_model_withL2 = vit.create_vit_classifier(num_classes,hp=hp_base_L2,regularization = 'L2', model_variant = model_name)
    print(realformer_model_withL2.summary())
    # keras.utils.plot_model(realformer_model_withL2,show_shapes=True,expand_nested=True,to_file="realformer_model_withL2.png")
    realformer_withL2_history = es.run_experiment(realformer_model_withL2, train_gen, validation_gen,model_name)
    plot_history(realformer_withL2_history,model=model,version=version)    
    
    ################# Augmeneted Data #################
    model_name = 'realformer_augmented'
    version='L2 reg aug'
    hp_augmented = hp_cls.Hyperparameters(image_size = 128,patch_size = 16,learning_rate = 0.01,batch_size=16)
    realformer_model_withL2_aug = vit.create_vit_classifier(num_classes,hp=hp_augmented,regularization = 'L2', model_variant = model_name)
    print(realformer_model_withL2_aug.summary())
    # keras.utils.plot_model(realformer_model_withL2_aug,show_shapes=True,expand_nested=True,to_file="realformer_model_withL2_aug.png")
    realformer_withL2_aug_history = es.run_experiment(realformer_model_withL2_aug, train_gen_aug, validation_gen_aug,model_name)
    plot_history(realformer_withL2_aug_history,model=model,version=version)
    
################# TESTING #################
#funtion to load the pre-trained models and test
def Realformer_Test(num_classes,test_gen, test_gen_aug):
    model='realformer'
    version='base'
    model_name = 'realformer'
    
    ################# BASE #################
    hp_base = hp_cls.Hyperparameters(image_size = 128,patch_size = 32)
    realformer_model = vit.create_vit_classifier(num_classes,hp=hp_base, model_variant = model_name)
    print('\n\n************ Base ************')
    es.test_model(realformer_model, test_gen,model_name,hp_base)
    
    ################# L2 REG #################
    model_name = 'realformer_l2'
    version='L2 reg'
    hp_base_L2 = hp_cls.Hyperparameters(image_size = 72,patch_size = 32,learning_rate = 0.0001)
    realformer_model_withL2 = vit.create_vit_classifier(num_classes,hp=hp_base_L2,regularization = 'L2', model_variant = model_name)
    print('\n\n************ L2 regularization ************')
    es.test_model(realformer_model_withL2, test_gen,model_name,hp_base_L2)
    
    ################# Augmeneted Data #################
    model_name = 'realformer_augmented'
    version='L2 reg aug'
    hp_augmented = hp_cls.Hyperparameters(image_size = 128,patch_size = 16,learning_rate = 0.01,batch_size=16)
    realformer_model_withL2_aug = vit.create_vit_classifier(num_classes,hp=hp_augmented,regularization = 'L2', model_variant = model_name)
    
    print('\n\n************ Data Without Augmentation ************')
    es.test_model(realformer_model_withL2_aug, test_gen,model_name,hp_augmented)
    print('\n\n************ Data With Augmentation ************')
    es.test_model(realformer_model_withL2_aug, test_gen_aug,model_name,hp_augmented)