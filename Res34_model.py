import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import tensorflow as tf
import pickle

from tensorflow.keras import backend as K

from tensorflow.keras.layers import Input, Dense, Lambda, Layer, Add, Multiply, add, Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.regularizers import l2


latent_dim = 128
  
  
def conv2d_bn(x,
              filters,
              strides,
              padding='same'
              ):

    x = tf.keras.layers.Conv2D(filters, (3, 3),
        strides=strides,
        padding=padding, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4),
        use_bias=False)(x)

    bn_axis = -1
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, scale=False)(x)
    x = tf.keras.layers.Activation('relu')(x)

    return x
    

def conv2d_bn_1x1(x,
              filters,
              strides,
              padding='same'
              ):

    x = tf.keras.layers.Conv2D(filters, (1, 1),
        strides=strides,
        padding=padding, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4),
        use_bias=False)(x)

    bn_axis = -1
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, scale=False)(x)
    x = tf.keras.layers.Activation('relu')(x)

    return x


def conv2d_bn_7x7(x,
              filters,
              strides=(2, 2),
              padding='same'
              ):
			  
    x = tf.keras.layers.Conv2D(filters, (7, 7),
        strides=strides,
        padding=padding, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4),
        use_bias=False)(x)

    bn_axis = -1
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, scale=False)(x)
    x = tf.keras.layers.Activation('relu')(x)

    return x
	
	
def residual_module1(layer_in, n_filters):
    # conv1
    x = conv2d_bn(layer_in, n_filters, strides=(1, 1), padding='same')

    # conv2
    conv2 = conv2d_bn(x, n_filters, strides=(1, 1), padding='same')

    # add filters, assumes filters/channels last
    layer_out = add([conv2, layer_in])
    # activation function
    layer_out = Activation('relu')(layer_out)

    return layer_out


def residual_module2(layer_in, n_filters):
    # conv1
    x = conv2d_bn(layer_in, n_filters, strides=(2, 2), padding='same')
    # conv2
    conv2 = conv2d_bn(x, n_filters, strides=(1, 1), padding='same')
    
    #projection shortcut for mismatch in number of channels
    y = conv2d_bn_1x1(layer_in, n_filters, strides=(2, 2), padding='same')

    # add filters, assumes filters/channels last
    layer_out = add([conv2, y])
    # activation function
    layer_out = Activation('relu')(layer_out)

    return layer_out
	
	
def res34(inp):
    channel_axis = -1
    #Instantiates the Inflated 3D Inception v1 architecture.

    # Downsampling via convolution (spatial and temporal)
    x = conv2d_bn_7x7(inp, 64, strides=(2, 2), padding='same')
    x = tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same')(x)
	
    x = residual_module1(x,64)
    x = residual_module1(x,64)
    x = residual_module1(x,64)

    x = residual_module2(x,128)
    x = residual_module1(x,128)
    x = residual_module1(x,128)
    x = residual_module1(x,128)

    x = residual_module2(x,256)
    x = residual_module1(x,256)
    x = residual_module1(x,256)
    x = residual_module1(x,256)
    x = residual_module1(x,256)
    x = residual_module1(x,256)

    x = residual_module2(x,512)
    x = residual_module1(x,512)
    x = residual_module1(x,512)
	
    x = tf.keras.layers.GlobalAveragePooling2D(name = 'gap')(x)
	
    # FCN with Relu activation, kernel_regularizer=l2(1e-3)
    x = tf.keras.layers.Dense(units=512, kernel_regularizer=l2(1e-3), name='dense1')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    # FCN with Relu activation, kernel_regularizer=l2(1e-3)
    x = tf.keras.layers.Dense(units=256, kernel_regularizer=l2(1e-3), name='dense2')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    z = tf.keras.layers.Dense(units = latent_dim)(x)
    z = tf.keras.layers.Activation('relu')(z)
    z = tf.keras.layers.Lambda(lambda x: K.l2_normalize(x, axis=1))(z)

    return z
	
def model_create():
    # defining input shape
    input = tf.keras.Input(shape=(121, 145, 1))

    # base for encoder
    z = res34(input)

    # defining model for encoder
    encoder = Model(inputs = input, outputs = z, name='encoder')
    print(encoder.summary())

    return encoder
