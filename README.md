With a lot of experiments conducted for image classification, this project aims to apply deep learning technology for Traffic Sign Detection. Although convolutional models are the primary source of the application, with increased attention towards ’attention layer-based Transformers’, in this project, dif- ferent versions of Transformer models are replicated and trained, to be compared against very deep convolutional layer network VGG19. Conducting experiments with 2 versions of Transformer namely ViT and Realformer against VGG19, transformers have been observed to be doing better with time and performance than standard convolutional networks for the given small dataset.

Requirements:
tensorflow-addons :- pip install tensorflow-addons

To execute the program: 
  1. Make sure the Main.py file has execute mode
  2. Command: ./Main.py test all > all_test.txt
  3. 1st argument takes in either train or test to train the models from scratch or to test a subset of data by loading the pre-trained weights.
  4. 2nd argument takes in vit or realformer or vgg or all implying the corresponding models or all together.
  5. cat all_test.txt | grep '************' ==> this command is to collect the testing result from the corresponding file.
  
  Project structure:
  Models - contains the the model structure expanded version for all 3 models
  training_2 - folder to hold the pre-trained model weights(on running new train saves the weights here)
  

 Dataset used in the project is available at https://www.kaggle.com/datasets/ahemateja19bec1025/traffic-sign-dataset-classification. On downloading place the data directly into Project in the following structure
  1. Data/labels.csv
  2. Data/traffic_Data/TRAIN
  3. Data/traffic_Data/TEST
 
 Download pre-trained model weights from : https://drive.google.com/file/d/1ItCviv-vvmWiWkSCV4vjZd2nyqE-jQS8/view?usp=share_link
 Unzip and place under Project
