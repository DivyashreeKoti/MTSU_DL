#!/usr/bin/env python3
import tensorflow.keras as keras
import matplotlib.pyplot as plt


import HyperParameters as hp_cls
import ExperimentSetup as es
import ConvolutionalNN as cnn
import Plot_patch as pp

################# PLOT Accuracy and Loss #################
def plot_history(history,model='vgg19',version='base'):
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
def VGG19(num_classes,train_gen,validation_gen,train_gen_aug,validation_gen_aug):
    model='VGG19'
    version='base'
    model_name = 'VGG19_base'
    
    hp_base = hp_cls.Hyperparameters(image_size=256,batch_size=16,learning_rate = 0.00001,
                weight_decay = 0.0000001)
    vgg = cnn.ConvolutionalNN(hp = hp_base)
    vgg19_model = vgg.create_cnn_model(num_classes)
    print(vgg19_model.summary())
    # keras.utils.plot_model(vgg19_model,show_shapes=True,expand_nested=False,to_file="vgg19_model.png")
    vgg19_history = es.run_experiment(vgg19_model, train_gen, validation_gen,model_name,hp=hp_base)
    plot_history(vgg19_history,model=model,version=version)
    
#     pp.plot_patch(train_gen,hp_base,model,version)
    
#     model_name = 'vgg19_augmented'
#     version='aug'
#     hp_augmented = hp_cls.Hyperparameters(image_size = 256,batch_size=64,learning_rate = 0.0001,
#                  weight_decay = 0.00001)
#     vgg = cnn.ConvolutionalNN(hp = hp_augmented)
#     vgg19_model_aug = vgg.create_cnn_model(num_classes)
#     vgg19_aug_history = es.run_experiment(vgg19_model_aug, train_gen_aug, validation_gen_aug,model_name,hp=hp_augmented)
#     plot_history(vgg19_aug_history,model=model,version=version)

################# TESTING #################
def VGG19_Test(num_classes,test_gen,test_gen_aug):
    model='VGG19'
    version='base'
    model_name = 'VGG19_base'
    
    hp_base = hp_cls.Hyperparameters(image_size=256,batch_size=16,learning_rate = 0.00001,
                weight_decay = 0.0000001)
    vgg = cnn.ConvolutionalNN(hp = hp_base)
    vgg19_model = vgg.create_cnn_model(num_classes)
    
    print('\n\n************ Data Without Augmentation ************')
    es.test_model(vgg19_model, test_gen,model_name,hp_base)
    
