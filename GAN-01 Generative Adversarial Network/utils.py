# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:08:31 2020

@author: jimhs
"""

import os
import matplotlib.pyplot as plt
import tensorflow as tf
from args import parser

opts = parser()

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    # total_loss = real_loss + fake_loss
    return real_loss, fake_loss


def generate_and_save_images(Generator, epoch, test_input):
    # Initial Log File
    fig_path = os.path.join(opts.PLOT_PATH)
    if not os.path.exists(fig_path):
        os.mkdir(fig_path)
        
    predictions = Generator(test_input, training=False)
    
    fig = plt.figure(figsize=(16,16))
    
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    fig.suptitle('{}'.format(epoch), fontsize=70)
    path = os.path.join(fig_path, '{:03d}.png'.format(epoch))
    plt.savefig(path)
    plt.close(fig)




















