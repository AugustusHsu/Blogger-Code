# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:08:31 2020

@author: jim
"""

import os
import imageio
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from args import parser

opts = parser()

cross_entropy = tf.keras.losses.BinaryCrossentropy()

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
    image = np.uint8(predictions*127.5 + 127.5)
    image = np.squeeze(image)
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(image[i], cmap='gray')
        plt.axis('off')
    fig.suptitle('{}'.format(epoch), fontsize=70)
    path = os.path.join(fig_path, '{:03d}.png'.format(epoch))
    plt.savefig(path)
    plt.close(fig)

def plot_GIF(anim_file = 'gan.gif'):
    images = []
    img_path = os.path.join(opts.PLOT_PATH)
    for epoch in range(opts.epochs):
        img = imageio.imread(os.path.join(img_path, '{:03d}.png'.format(epoch+1)))
        images.append(img)
    imageio.mimsave(anim_file, images)

def plot_line(df, col_name, method, figname):
    # Initial Log File
    fig_path = os.path.join(opts.PLOT_LINE_PATH)
    if not os.path.exists(fig_path):
        os.mkdir(fig_path)
        
    fig, ax = plt.subplots(figsize=(16,10))
    sns.lineplot(data=df[col_name])
    
    fig.suptitle(figname, fontsize=50)
    path = os.path.join(fig_path, method + '_' + figname + '.png')
    plt.savefig(path)
    plt.close('all')
















