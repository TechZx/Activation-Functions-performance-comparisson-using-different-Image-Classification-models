##################################################################################################
# Author: George Zacharis
##################################################################################################



![preview](https://github.com/TechZx/Activation-Functions-performance-comparisson-using-different-Image-Classification-models/assets/43382759/a6d882cb-e386-4c2a-9dd1-c190de23f25a)




# Activation-Functions-performance-comparison-using-different-Image-Classification-Models
The following project is to test the performance of different pretrained deep learning models while changing the activation functions of the original layers. The task is to observe any improvements in performance or training time. The performance will be tested at image classification tasks and with different datasets.  

Pretrained models : Xception,InceptionV3,DenseNet121,ResNet50V2,EfficientNetV2B0,MobileNetV2

Datasets: Cifar10,cifar100,stl10,svhn,stl10

Activation Functions : ReLU,Sigmoid,Tanh,ELU,GELU,SELU,Swish,Mish,SMish


# TO RUN 

0. Install anaconda for easier installation of dependencies (recommended)
1. Install dependencies to run (Tensorflow, TFDS, KERAS, numpy, pandas)
2. Install extra_keras_datasets
3. Required python 3 (version depending on TF and CUDA Versions)
4. Install wandb to save results and graphs
5. Change wandb repository to the oner you created to pass the data for graph extraction (wandb is optional)

# NOTES

- Can modify input dataset images size other than default 75x75 (might require big amounts of processing power)
- Hyperparameters can be tuned in the first lines of autopilot.py for experimentation or better performance
- For loop for autorun may require big amounts of resources. In such case reduce for loops or run each for loop separately
- processes.py can be modified for more activation functions (for more see https://www.tensorflow.org/api_docs/python/tf/keras/activations)
- The results and graphs shown are from personal experiments
