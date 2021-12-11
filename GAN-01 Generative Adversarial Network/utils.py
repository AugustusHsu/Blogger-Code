# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2021/11/30 17:11:16
@Author  :   AugustusHsu
@Contact :   jimhsu11@gmail.com
'''

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
        
    fig = plt.figure(figsize=(16,16))
    image = np.uint8(pred_img*127.5 + 127.5)
    image = np.squeeze(image)
    for i in range(pred_img.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(image[i], cmap='gray')
        plt.axis('off')
    fig.suptitle('{}'.format(epoch), fontsize=70)
    path = os.path.join(fig_path, '{:03d}.png'.format(epoch))
    plt.savefig(path)
    plt.close(fig)

def plot_GIF(anim_file='gan.gif'):
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
