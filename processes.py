import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.layers import Activation
from tensorflow.keras import activations
from tensorflow.keras.models import Model
from ActTensor_tf import Smish

 
################################################################################################
def print_stats(num_classes,btch,lr,epchs):
    print("################################")
    print("Number of classes: \t"  +str(num_classes))
    print("Batch Size : \t\t" +str(btch))
    print("Learning Rate : \t" +str(lr))
    print("Number of Epochs : \t" +str(epchs))
    print("################################")
################################################################################################
    
################################################################################################    
def preprocessing(x_train,x_test,y_train,y_test,preprocess_input,num_classes):
  
    # Add a new dimension for the channels
    # x_train = tf.expand_dims(x_train, axis=-1)
    # x_test = tf.expand_dims(x_test, axis=-1)

    # y_train to categorical
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # Rescale pixel values to between 0 and 1 (xtrain_rescaled)
    x_train = x_train.astype('float32') / 127.5
    x_test = x_test.astype('float32') / 127.5
    print('################ Data Rescaling Successful ################')

    # Resize images to 75x75 pixels (xtrain_resized)
    x_train = tf.image.resize(x_train, [75, 75])
    x_test = tf.image.resize(x_test, [75, 75])
    print('################ Data Resizing Successful ################')

    # Preprocess images using the preprocess_input of specific model
    x_train = preprocess_input(x_train)
    x_test = preprocess_input(x_test)
    print('################ Data Preprocessing Successful ################')
    return x_train,x_test,y_train,y_test
################################################################################################

################################################################################################
def change_activation(act, model):
    c = 0
    for layer in model.layers:
        if hasattr(layer, 'activation'):
            if act == "relu":
                layer.activation = tf.keras.activations.relu
            elif act == "sigmoid":
                layer.activation = tf.keras.activations.sigmoid
            elif act == "tanh":
                layer.activation = tf.keras.activations.tanh
            elif act == "elu":
                layer.activation = tf.keras.activations.elu
            elif act == "gelu":
                layer.activation = tf.keras.activations.gelu
            elif act == "selu":
                layer.activation = tf.keras.activations.selu
            elif act == "swish":
                layer.activation = tf.keras.activations.swish
            elif act == "mish":
                layer.activation = tf.keras.activations.mish
            elif act == "smish":
                layer.activation = Smish() 
            elif act == "original":
                print("No activation functions were changed loading original model ...")
                break
            else:
                print("Invalid activation function")
                return
            c += 1
    if (c!=0):print("################ " + str(c) + " Activation functions changed to " + act + " Successfully ################")
    return model
################################################################################################

