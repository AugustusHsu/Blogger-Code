# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2021/11/30 17:11:16
@Author  :   AugustusHsu
@Contact :   jimhsu11@gmail.com
'''

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Flatten,
    Dense,
    Reshape,
    Input,
)
from absl.flags import FLAGS

def Dis_Net():
    x = inputs = Input([28, 28, 1])
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)
    return Model(inputs, output, name='Discriminator')
        
def Gen_Net():
    x = inputs = Input([FLAGS.noise_dim])
    x = Dense(128, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(28 * 28 * 1, activation='tanh')(x)
    output = Reshape((28, 28, 1))(x)
    return Model(inputs, output, name='Generator')
