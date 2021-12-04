# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2021/11/30 17:11:16
@Author  :   AugustusHsu
@Contact :   jimhsu11@gmail.com
'''

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Concatenate,
    Flatten,
    Dense,
    Dropout,
    Reshape,
    Input,
)
from absl.flags import FLAGS

def Maxout(x, units, k, activation, drop_prob=0.5):
    x = Dense(k * units, activation=activation)(x)
    x = Dropout(drop_prob)(x)
    x = Reshape((k, units))(x)
    x = tf.reduce_max(x, axis=1)
    return x

def Dis_Net():
    x = inputs_x = Input([28, 28, 1])
    x = Flatten()(x)
    x = Maxout(x, units=240, k=5, activation='relu', drop_prob=0.5)
    
    y = labels_x = Input([FLAGS.num_classes])
    y = Maxout(y, units=50, k=5, activation='relu', drop_prob=0.5)
    
    x = Concatenate()([x, y])
    x = Maxout(x, units=240, k=4, activation='relu', drop_prob=0.5)
    
    output = Dense(1, activation='sigmoid')(x)
    
    return Model([inputs_x, labels_x], output, name='Discriminator')
        
def Gen_Net():
    x = inputs_z = Input([FLAGS.noise_dim])
    x = Dense(256, activation='relu')(x)
    
    y = labels_x = Input([FLAGS.num_classes])
    y = Dense(256, activation='relu')(y)
    
    x = Concatenate()([x, y])
    x = Dense(512, activation='relu')(x)
    x = Dense(28 * 28 * 1, activation='tanh')(x)
    output = Reshape((28, 28, 1))(x)
    
    return Model([inputs_z, labels_x], output, name='Generator')
