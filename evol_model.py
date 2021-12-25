'''
Creation of keras models based on chromosomes.
'''

import tensorflow as tf
import numpy as np
import random as rd
from utils import stride

kl = tf.keras.layers

def choose(category):
    if category in ("n_filter_a", "n_filter_b"):
        choice = rd.choice([8, 16, 32, 64])
    elif category in ("kernel_size_a", "kernel_size_b"):
        choice = rd.choice([1, 3, 5])
    elif category == "activation":
        choice = rd.choice(["relu", "tanh", "selu", "elu"])
    elif category == "pooling_type":
        choice = rd.choice(["max", "average"])
    elif category in ("do_skip", "do_BN_a", "do_BN_b", "do_pool"):
        choice = rd.randint(0,1)
    elif category == "n_layers":
        choice = rd.choice([1,2])
    else:
        raise NotImplementedError
    return choice


def create_model(chr):
    #Build model from chromosome
    Input = kl.Input(shape = (28,28, 1))
    block = 0
    while True:
        if block not in chr: break
        chrr = chr[block]
        n = 28

        if block == 0:
            #First conv layer:
            activation = chrr["activation"]
            x = kl.Conv2D(chrr["n_filter_a"], kernel_size = chrr["kernel_size_a"], activation = activation, padding = "same")(Input)
            #Pooling:
            if chrr["do_pool"]:
                n //= 2
                if chrr["pooling_type"] == "max":
                    x = kl.MaxPooling2D()(x)
                elif chrr["pooling_type"] == "average":
                    x = kl.AveragePooling2D()(x)
                else:
                    raise NotImplementedError
            #Batchnorm:
            if chrr["do_BN_a"]:
                x = kl.BatchNormalization()(x)
            
            #Second conv layer:
            n -= chrr["kernel_size_b"]-1
            x = kl.Conv2D(chrr["n_filter_b"], kernel_size = chrr["kernel_size_b"], activation = activation, padding = "same")(x)
            #Batchnorm:
            if chrr["do_BN_b"]:
                x = kl.BatchNormalization()(x)
            
            #Residual (n, n, n_filter_b) + transformation(28, 28, 1)
            if chrr["do_skip"]:
                s = stride(n_in = 28, n_out = n, k = 1, p = 0)
                Input_transformed = kl.Conv2D(chrr["n_filter_b"], kernel_size = 1, padding = "same", strides = (s,s))(Input)
                x = kl.Add()([x, Input_transformed])

            #Final activation
            x = tf.keras.layers.Activation(activation = activation)(x)
            
        else:
            pass

        block += 1
    
    x = kl.GlobalMaxPooling2D()(x)
    x = kl.Flatten()(x)
    outputs = kl.Dense(10, activation = "softmax")(x)
    
    return tf.keras.Model(inputs = Input, outputs = outputs)