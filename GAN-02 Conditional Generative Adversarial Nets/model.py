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
    Reshape,
    Input,
)
from absl import flags
from absl.flags import FLAGS

flags.DEFINE_boolean('EachLayer', False, 'each layer add the label info')

def Maxout(inputs, num_units, axis=-1):
    # this code is from TensorFlow addons' code:
    # https://github.com/tensorflow/addons/blob/v0.15.0/tensorflow_addons/layers/maxout.py
    shape = inputs.get_shape().as_list()
    
    # Dealing with batches with arbitrary sizes
    for i in range(len(shape)):
        if shape[i] is None:
            shape[i] = tf.shape(inputs)[i]
            
    num_channels = shape[axis]
    if axis < 0:
        axis = axis + len(shape)
    assert axis >= 0, "Find invalid axis: {}".format(axis)
    
    expand_shape = shape[:]
    expand_shape[axis] = num_units
    k = num_channels // num_units
    expand_shape.insert(axis, k)

    outputs = tf.math.reduce_max(
        tf.reshape(inputs, expand_shape), axis, keepdims=False
    )
    return outputs

def Dis_Net():
    x = inputs_x = Input([28, 28, 1])
    x = Flatten()(x)
    x = Dense(240, activation='linear')(x)
    x = Maxout(x, 5)
    
    y = labels_x = Input([10])
    y = Dense(50, activation='linear')(y)
    y = Maxout(y, 5)
    
    x = Concatenate()([x, y])
    x = Dense(240, activation='linear')(x)
    x = Maxout(x, 4)
    output = Dense(1, activation='sigmoid')(x)
    
    return Model([inputs_x, labels_x], output, name='Discriminator')
        
def Gen_Net():
    x = inputs_z = Input([FLAGS.noise_dim])
    x = Flatten()(x)
    y = labels_x = Input([10])
    x = Dense(128, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(28 * 28 * 1, activation='tanh')(x)
    output = Reshape((28, 28, 1))(x)
    return Model(inputs, output, name='Generator')

# #定义batchnorm(批次归一化)层
# def batch_norm(input_, name="batch_norm"):
#     with tf.variable_scope(name):
#         input_dim = input_.get_shape()[-1]
#         scale = tf.get_variable("scale", [input_dim], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
#         offset = tf.get_variable("offset", [input_dim], initializer=tf.constant_initializer(0.0))
#         mean, variance = tf.nn.moments(input_, axes=[1,2], keep_dims=True)
#         epsilon = 1e-5
#         inv = tf.rsqrt(variance + epsilon)
#         normalized = (input_-mean)*inv
#         output = scale*normalized + offset
#         return output
