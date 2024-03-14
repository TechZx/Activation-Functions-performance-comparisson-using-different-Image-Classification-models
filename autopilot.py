##################################################################################################

# Autorun script for testing 
# Author: George Zacharis
# Github: https://github.com/TechZx

##################################################################################################

#basic imports
import tensorflow as tf
from tensorflow import keras
import time
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from keras import initializers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D,GlobalAveragePooling2D
from tensorflow.python.ops.numpy_ops import np_config
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
from tensorflow.python.keras.layers import Activation
from tensorflow.keras import activations
from tensorflow.keras.optimizers import Adam
from processes import *
import wandb
from wandb.keras import WandbCallback
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
import tensorflow_datasets as tfds

# Tuning hyperparameters (number of classes,batch size, learning rate,epochs,activation function(optional))
num_classes = 10
lr=0.001
epchs=10
btch=64
act= ["original","relu","sigmoid","tanh","elu","gelu","selu","swish","mish","smish"]
wandb_project="test"

# Load the dataset for training
if wandb_project == "test" or wandb_project == "CIFAR-10":
    from tensorflow.keras.datasets import cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Load the CIFAR-100 dataset
elif wandb_project=="CIFAR-100":
    from tensorflow.keras.datasets import cifar100
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()

# Load the STL-10 dataset
elif wandb_project=="STL-10":
    from tensorflow.keras.datasets import stl10
    (x_train, y_train), (x_test, y_test) = stl10.load_data()

# Load the SVHN dataset
elif wandb_project=="SVHN":
    from extra_keras_datasets import svhn
    (x_train, y_train), (x_test, y_test) = svhn.load_data('normal')


def autorunXception(act,num_classes,btch,lr,epchs,x_train, y_train,x_test, y_test):
    #Xception imports
    from tensorflow.keras.applications import Xception
    from tensorflow.keras.applications.xception import preprocess_input
    tf.experimental.numpy.experimental_enable_numpy_behavior()
    # Wandb parameters
    # Weights and Biases config for graphs
    wandb.init(project=str(wandb_project), id="Xception " + str(act))

    # Data Preprocessing
    x_train,x_test,y_train,y_test=preprocessing(x_train,x_test,y_train,y_test,preprocess_input,num_classes)

    # Load the pre-trained Xception model
    # Due to the nature of the experiment we will not need the pretrained weights
    model = Xception(weights=None, include_top=False,input_shape=(75,75,3))

    # Uncomment to change activation functions except for using original network
    # Change the activation functions of the model into the one tested each time
    change_activation(act,model)
    # Print the value of the hyperparameters
    print_stats(num_classes,btch,lr,epchs)

    # Adding a global average pooling layer and a fully connected layer
    x = model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    # base_model.summary()

    #Create the final model
    model = Model(inputs=model.input, outputs=predictions)

    start = time.perf_counter()
    #Compile the model
    model.compile(optimizer=Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    Xception = model.fit(x_train, y_train, epochs=epchs, batch_size=btch, validation_data=(x_test, y_test), callbacks=[WandbMetricsLogger(log_freq=5),WandbModelCheckpoint("models")])

    elapsed = time.perf_counter() - start
    print('Elapsed %.3f seconds.' % elapsed)
    wandb.finish()

def autorunInceptionV3(act,num_classes,btch,lr,epchs,x_train, y_train,x_test, y_test):

    #InceptionV3 imports
    from tensorflow.keras.applications import InceptionV3
    from tensorflow.keras.applications.inception_v3 import preprocess_input
    np_config.enable_numpy_behavior()
    # Wandb parameters
    # Weights and Biases config for graphs
    wandb.init(project=str(wandb_project), id="Inception V3 " + str(act))

    # Data Preprocessing
    x_train,x_test,y_train,y_test=preprocessing(x_train,x_test,y_train,y_test,preprocess_input,num_classes)

    # Load the pre-trained InceptionV3 model
    # Due to the nature of the experiment we will not need the pretrained weights
    model = InceptionV3(weights=None, include_top=False,input_shape=(75,75,3))

    # Uncomment to change activation functions except for using original network
    # Change the activation functions of the model into the one tested each time
    change_activation(act,model)
    # Print the value of the hyperparameters
    print_stats(num_classes,btch,lr,epchs)

    # Adding a global average pooling layer and a fully connected layer
    x = model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    # base_model.summary()

    #Create the final model
    model = Model(inputs=model.input, outputs=predictions)

    start = time.perf_counter()
    #Compile the model
    model.compile(optimizer=Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    InceptionV3 = model.fit(x_train, y_train, epochs=epchs, batch_size=btch, validation_data=(x_test, y_test), callbacks=[WandbMetricsLogger(log_freq=5),WandbModelCheckpoint("models")])
    elapsed = time.perf_counter() - start
    print('Elapsed %.3f seconds.' % elapsed)
    wandb.finish()

def autorunDenseNet121(act,num_classes,btch,lr,epchs,x_train,y_train,x_test, y_test):
    # DenseNet121 imports
    from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input

    # Wandb parameters
    # Weights and Biases config for graphs
    wandb.init(project=str(wandb_project), id="DenseNet121 " + str(act))

    # Data Preprocessing
    x_train,x_test,y_train,y_test=preprocessing(x_train,x_test,y_train,y_test,preprocess_input,num_classes)

    # Load the pre-trained DenseNet121 model
    # Due to the nature of the experiment we will not need the pretrained weights
    model = DenseNet121(weights=None, include_top=False,input_shape=(75,75,3))

    # Uncomment to change activation functions except for using original network
    # Change the activation functions of the model into the one tested each time
    change_activation(act,model)
    # Print the value of the hyperparameters
    print_stats(num_classes,btch,lr,epchs)

    # Adding a global average pooling layer and a fully connected layer
    x = model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    # base_model.summary()

    #Create the final model
    model = Model(inputs=model.input, outputs=predictions)

    start = time.perf_counter()
    #Compile the model
    model.compile(optimizer=Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    DenseNet121 = model.fit(x_train, y_train, epochs=epchs, batch_size=btch, validation_data=(x_test, y_test), callbacks=[WandbMetricsLogger(log_freq=5),WandbModelCheckpoint("models")])
    elapsed = time.perf_counter() - start
    print('Elapsed %.3f seconds.' % elapsed)
    wandb.finish()

def autorunEfficientNetV2B0(act,num_classes,btch,lr,epchs,x_train, y_train,x_test, y_test):
    # EfficientNetV2B0 imports
    from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B0,preprocess_input
    tf.experimental.numpy.experimental_enable_numpy_behavior()
    # Wandb parameters
    # Weights and Biases config for graphs
    wandb.init(project=str(wandb_project), id="EfficientNet V2B0 " + str(act))

    # Data Preprocessing
    x_train,x_test,y_train,y_test=preprocessing(x_train,x_test,y_train,y_test,preprocess_input,num_classes)

    # Load the pre-trained EfficientNetV2B0 model
    # Due to the nature of the experiment we will not need the pretrained weights
    model = EfficientNetV2B0(weights=None, include_top=False,input_shape=(75,75,3))

    # Uncomment to change activation functions except for using original network
    # Change the activation functions of the model into the one tested each time
    change_activation(act,model)
    # Print the value of the hyperparameters
    print_stats(num_classes,btch,lr,epchs)

    # Adding a global average pooling layer and a fully connected layer
    x = model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    # base_model.summary()

    #Create the final model
    model = Model(inputs=model.input, outputs=predictions)

    start = time.perf_counter()
    #Compile the model
    model.compile(optimizer=Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    EfficientNetV2B0 = model.fit(x_train, y_train, epochs=epchs, batch_size=btch, validation_data=(x_test, y_test), callbacks=[WandbMetricsLogger(log_freq=5),WandbModelCheckpoint("models")])
    elapsed = time.perf_counter() - start
    print('Elapsed %.3f seconds.' % elapsed)
    wandb.finish()

def autorunMobileNetV2(act,num_classes,btch,lr,epchs,x_train, y_train,x_test, y_test):
    # MobileNetV2 imports
    from tensorflow.keras.applications import MobileNetV2 
    from tensorflow.keras.applications.mobilenet_v2 import  preprocess_input
    
    tf.experimental.numpy.experimental_enable_numpy_behavior()
    # Wandb parameters
    # Weights and Biases config for graphs
    wandb.init(project=str(wandb_project), id="MobileNetV2 " + str(act))

    # Data Preprocessing
    x_train,x_test,y_train,y_test=preprocessing(x_train,x_test,y_train,y_test,preprocess_input,num_classes)

    # Load the pre-trained MobileNetV2 model
    # Due to the nature of the experiment we will not need the pretrained weights
    model = MobileNetV2(weights=None, include_top=False,input_shape=(75,75,3))

    # Uncomment to change activation functions except for using original network
    # Change the activation functions of the model into the one tested each time
    change_activation(act,model)
    # Print the value of the hyperparameters
    print_stats(num_classes,btch,lr,epchs)

    # Adding a global average pooling layer and a fully connected layer
    x = model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    # base_model.summary()

    #Create the final model
    model = Model(inputs=model.input, outputs=predictions)

    start = time.perf_counter()
    #Compile the model
    model.compile(optimizer=Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    MobileNetV2 = model.fit(x_train, y_train, epochs=epchs, batch_size=btch, validation_data=(x_test, y_test), callbacks=[WandbMetricsLogger(log_freq=5),WandbModelCheckpoint("models")])
    elapsed = time.perf_counter() - start
    print('Elapsed %.3f seconds.' % elapsed)
    wandb.finish()

def autorunResNet50V2(act,num_classes,btch,lr,epchs,x_train, y_train,x_test, y_test):
    # ResNet50V2 imports
    from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input
    tf.experimental.numpy.experimental_enable_numpy_behavior()
    # Wandb parameters
    # Weights and Biases config for graphs
    wandb.init(project=str(wandb_project), id="ResNet50V2 " + str(act))

    # Data Preprocessing
    x_train,x_test,y_train,y_test=preprocessing(x_train,x_test,y_train,y_test,preprocess_input,num_classes)

    # Load the pre-trained ResNet50V2 model
    # Due to the nature of the experiment we will not need the pretrained weights
    model = ResNet50V2(weights=None, include_top=False,input_shape=(75,75,3))

    # Uncomment to change activation functions except for using original network
    # Change the activation functions of the model into the one tested each time
    change_activation(act,model)
    # Print the value of the hyperparameters
    print_stats(num_classes,btch,lr,epchs)

    # Adding a global average pooling layer and a fully connected layer
    x = model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    # base_model.summary()

    #Create the final model
    model = Model(inputs=model.input, outputs=predictions)

    start = time.perf_counter()
    #Compile the model
    model.compile(optimizer=Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    ResNet50V2 = model.fit(x_train, y_train, epochs=epchs, batch_size=btch, validation_data=(x_test, y_test), callbacks=[WandbMetricsLogger(log_freq=5),WandbModelCheckpoint("models")])
    elapsed = time.perf_counter() - start
    print('Elapsed %.3f seconds.' % elapsed)
    wandb.finish()



# runs
for x in range(6):

    # Xception
    for i in range (10):    
        autorunXception(act[i],num_classes,btch,lr,epchs,x_train, y_train,x_test, y_test)
        print("################ Xception " + str(act[i])+ " run completed Successfully ################")

    # InceptionV3
    for i in range (10):
        autorunInceptionV3(act[i],num_classes,btch,lr,epchs,x_train, y_train,x_test, y_test)
        print("################ InceptionV3 " + str(act[i])+ " run completed Successfully ################")

     # DenseNet121
    for i in range (10):
        act= ["original","relu","sigmoid","tanh","elu","gelu","selu","swish","mish","smish"]
        autorunDenseNet121(act[i],num_classes,btch,lr,epchs,x_train, y_train,x_test, y_test)
        print("################ DenseNet121 " + str(act[i])+ " run completed Successfully ################")

     # EfficientNetV2B0
    for i in range (10):
        autorunEfficientNetV2B0(act[i],num_classes,btch,lr,epchs,x_train, y_train,x_test, y_test)
        print("################ EfficientNetV2B0 " + str(act[i])+ " run completed Successfully ################")

     # MobileNetV2
    for i in range (10):
        act= ["original","relu","sigmoid","tanh","elu","gelu","selu","swish","mish","smish"]
        autorunMobileNetV2(act[i],num_classes,btch,lr,epchs,x_train, y_train,x_test, y_test)
        print("################ MobileNetV2 " + str(act[i])+ " run completed Successfully ################")
 
    # ResNet50V2
    for i in range (10):
        autorunResNet50V2(act[i],num_classes,btch,lr,epchs,x_train, y_train,x_test, y_test)
        print("################ ResNet50V2 " + str(act[i])+ " run completed Successfully ################")




