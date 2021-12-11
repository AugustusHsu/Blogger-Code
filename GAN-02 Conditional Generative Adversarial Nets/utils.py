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

from absl import flags
from absl.flags import FLAGS

flags.DEFINE_string('PLOT_PATH', 'plot/', 
                    'path to save the temporary image')
flags.DEFINE_string('IMG_PATH', 'img/', 
                    'path to save the final image')

cross_entropy = tf.keras.losses.BinaryCrossentropy()

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    # total_loss = real_loss + fake_loss
    return real_loss, fake_loss

def generate_and_save_images(pred_img, epoch):
    # Create folder
    fig_path = os.path.join(FLAGS.PLOT_PATH)
    if not os.path.exists(fig_path):
        os.mkdir(fig_path)
    
    images = np.squeeze(pred_img)
    reshaped_images = np.reshape(images, [20, 10]+list(images.shape)[1:])
    class_images = np.concatenate([image for image in reshaped_images], axis=2)
    example_images = np.concatenate([image for image in class_images], axis=0)
    save_image = np.uint8(example_images*127.5 + 127.5)
    
    fig = plt.figure(figsize=(16,9), dpi=80)
    fig.suptitle('{}'.format(epoch), fontsize=50)
    plt.imshow(save_image, cmap='gray')
    path = os.path.join(fig_path, '{:03d}.png'.format(epoch))
    plt.savefig(path, bbox_inches='tight')
    plt.close(fig)

def plot_GIF(anim_file='cgan.gif'):
    if not os.path.exists(FLAGS.IMG_PATH):
        os.mkdir(FLAGS.IMG_PATH)
    images = []
    img_path = os.path.join(FLAGS.PLOT_PATH)
    for epoch in range(FLAGS.epochs):
        img = imageio.imread(os.path.join(img_path, '{:03d}.png'.format(epoch+1)))
        images.append(img)
    imageio.mimsave(os.path.join(FLAGS.IMG_PATH, anim_file), images)

def plot_line(df, col_name, method, figname):
    # Create folder
    fig_path = os.path.join(FLAGS.IMG_PATH)
    if not os.path.exists(fig_path):
        os.mkdir(fig_path)
        
    fig, ax = plt.subplots(figsize=(16,10))
    sns.lineplot(data=df[col_name])
    
    fig.suptitle(figname, fontsize=50)
    path = os.path.join(fig_path, method + '_' + figname + '.png')
    plt.savefig(path)
    plt.close('all')
