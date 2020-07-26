# -*- coding: utf-8 -*-
"""
Created on Wed May 13 14:57:25 2020

@author: jimhs
"""

import tensorflow as tf
from tensorflow.keras import Model, layers

class Dis_Net(Model):
    def __init__(self):
        super(Dis_Net, self).__init__(self)
        
        self.Dense = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(1)])
    def call(self, vec):
        output = self.Dense(vec)
        return output
        
class Gen_Net(Model):
    def __init__(self, channels=1):
        super(Gen_Net, self).__init__(self)
        self.channels = channels
        
        self.Dense = tf.keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(512, activation='relu'),
            layers.Dense(28 * 28 * self.channels, activation='tanh'),
            layers.Reshape((28, 28, self.channels))])
    def call(self, vec):
        output = self.Dense(vec)
        return output








