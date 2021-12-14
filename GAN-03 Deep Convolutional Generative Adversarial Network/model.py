# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2021/12/07 18:45:46
@Author  :   AugustusHsu
@Contact :   jimhsu11@gmail.com
'''

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Conv2D,
    Flatten,
    Dense,
    LeakyReLU,
    BatchNormalization,
    Input,
    Conv2DTranspose,
    Reshape,
    Dropout,
)
from absl.flags import FLAGS

def Dis_Net(ImageSize, Channel):
    # padding='same':
    #   shape = [flood(Length/strides[0]), flood(Width/strides[1]), filters]
    # padding='valid':
    #   shape = [ceil((Length - kernel_size[0] + 1)/strides[0]),
    #            ceil((Width - kernel_size[1] + 1)/strides[1]), filters]
    Length, Width = ImageSize
    x = inputs_x = Input([Length, Width, Channel])
    x = Conv2D(128, (4, 4), strides=(2, 2), padding='same')(x)
    x = LeakyReLU()(x)
    
    x = Conv2D(256, (4, 4), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(0.3)(x)
    
    x = Conv2D(512, (4, 4), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(0.3)(x)
    
    x = Conv2D(1024, (4, 4), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(0.3)(x)
    
    x = Conv2D(1, (4, 4), strides=(1, 1), padding='valid')(x)
    x = Dropout(0.3)(x)
    
    x = Flatten()(x)
    x = Dense(1)(x)
    output = tf.sigmoid(x)
    return Model(inputs_x, output, name='Discriminator')

def Gen_Net(Channel):
    # padding_array(將input做padding的陣列):
    #   shape = [input_size - 1]*stride + 2*(kernel_size - 1) + 1
    # 將padding_image做conv2d valid padding的卷積，得到公式如下:
    #    padding='same':
    #       shape = [Length*strides[0], Width*strides[1], filters]
    #    padding='valid':
    #       shape = [(Length - 1)*stride[0] + kernel_size[0],
    #               (Width - 1)*stride[1] + kernel_size[1], filters]
    
    x = inputs_x = Input([FLAGS.noise_dim])
    x = Reshape((1, 1, FLAGS.noise_dim))(x)
    
    x = Conv2DTranspose(1024, (4, 4), strides=(1, 1), padding='valid')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    x = Conv2DTranspose(512, (4, 4), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    x = Conv2DTranspose(256, (4, 4), strides=(2, 2),  padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    x = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    output = Conv2DTranspose(Channel, (4, 4), strides=(2, 2), padding='same', activation='tanh')(x)
    return Model(inputs_x, output, name='Generator')