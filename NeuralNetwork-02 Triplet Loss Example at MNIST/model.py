# -*- coding: utf-8 -*-
"""
Created on Thu May 28 04:02:25 2020

@author: jimhs
"""

import tensorflow as tf
from tensorflow.keras import Model, layers

class Triplet_Net(Model):
    def __init__(self):
        super(Triplet_Net, self).__init__(self)
        self.emb_layer = tf.keras.models.Sequential([
            layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'),
            layers.LeakyReLU(),
            layers.Dropout(0.3),
            
            layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'),
            layers.LeakyReLU(),
            layers.Dropout(0.3),
            
            layers.Flatten(),
            tf.keras.layers.Dense(200)])
        self.l2_norm = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
    def call(self, img, training=True):
        vec = self.emb_layer(img)
        vec = self.l2_norm(vec)
        return vec

