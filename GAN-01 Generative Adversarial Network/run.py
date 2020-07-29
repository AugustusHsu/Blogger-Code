# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:00:16 2020

@author: jimhs
"""
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.datasets.mnist import load_data
from model import Dis_Net, Gen_Net
from utils import generator_loss
from utils import discriminator_loss
from utils import generate_and_save_images
from utils import plot_GIF

from tqdm import tqdm
from args import parser

opts = parser()
'''
-----------------------Data Set-----------------------
'''
[(train_x, train_y), (test_x, test_y)] = load_data('mnist.npz')

train_images = train_x.reshape(train_x.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5

train_dataset = tf.data.Dataset.from_tensor_slices(train_images)
train_dataset = train_dataset.shuffle(opts.BUFFER_SIZE)
train_dataset = train_dataset.batch(opts.BATCH_SIZE)

noise = tf.random.normal([opts.num_examples_to_generate, opts.noise_dim])
'''
-----------------------Network Setting-----------------------
'''
Generator = Gen_Net()
Discriminator = Dis_Net()

G_opt = tf.keras.optimizers.Adam(opts.lr, opts.beta_1)
D_opt = tf.keras.optimizers.Adam(opts.lr, opts.beta_1)

'''
-----------------------Each Epoch Training-----------------------
'''
@tf.function
def train_step(images, loss):
    noise = tf.random.normal([opts.BATCH_SIZE, opts.noise_dim])
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = Generator(noise, training=True)
        
        real_output = Discriminator(images, training=True)
        fake_output = Discriminator(generated_images, training=True)
        
        gen_loss = generator_loss(fake_output)
        real_loss, fake_loss = discriminator_loss(real_output, fake_output)
        disc_loss = real_loss + fake_loss
        
    loss[0].update_state(real_output)
    loss[1].update_state(fake_output)
    loss[2].update_state(gen_loss)
    loss[3].update_state(disc_loss)
    gradients_of_gen = gen_tape.gradient(gen_loss, Generator.trainable_variables)
    gradients_of_dis = disc_tape.gradient(disc_loss, Discriminator.trainable_variables)
    
    G_opt.apply_gradients(zip(gradients_of_gen, Generator.trainable_variables))
    D_opt.apply_gradients(zip(gradients_of_dis, Discriminator.trainable_variables))
    
'''
-----------------------Training-----------------------
'''
def train(train_dataset):
    # Initial Log File
    log_path = os.path.join(opts.LOG_PATH)
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    csv_path = os.path.join(log_path, 'loss.csv')
    with open(csv_path, 'w') as f:
        f.write('epoch,Real_P,Fake_P,Gen_loss,Dis_loss\n')
    format_str = '{:5d},{:.6f},{:.6f},{:.6f},{:.6f}\n'
    dis_r_p = tf.keras.metrics.Mean()
    dis_f_p = tf.keras.metrics.Mean()
    G_loss = tf.keras.metrics.Mean()
    D_loss = tf.keras.metrics.Mean()
    loss = [dis_r_p, dis_f_p, G_loss, D_loss]
    
    # Training
    for epoch in range(opts.epochs):
        start = time.time()
        for image_batch in tqdm(train_dataset.as_numpy_iterator()):
            train_step(image_batch, loss)
        
        # Record Loss
        with open(csv_path, 'a') as f:
            f.write(format_str.format(epoch,
                                      loss[0].result().numpy(),
                                      loss[1].result().numpy(),
                                      loss[2].result().numpy(),
                                      loss[3].result().numpy()))
        loss[0].reset_states()
        loss[1].reset_states()
        loss[2].reset_states()
        loss[3].reset_states()
        # Each Epoch Save Image
        generate_and_save_images(Generator,
                                 epoch + 1,
                                 noise)
        
        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            Gen_save_path = os.path.join(opts.MODEL_PATH, 'Generator')
            Dis_save_path = os.path.join(opts.MODEL_PATH, 'Discriminator')
            Generator.save_weights(Gen_save_path)
            Discriminator.save_weights(Dis_save_path)
        
        print ('Time for epoch {} is {:.3f} sec'.format(epoch + 1, time.time()-start))
        time.sleep(0.2)
    
    # Generate after the final epoch
    # display.clear_output(wait=True)
    # generate_and_save_images(Generator,
    #                          epochs,
    #                          noise)

train(train_dataset)
plot_GIF()








