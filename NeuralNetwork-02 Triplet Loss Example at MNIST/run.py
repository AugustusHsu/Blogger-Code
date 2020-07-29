# -*- coding: utf-8 -*-
"""
Created on Thu May 28 02:10:55 2020

@author: jimhs
"""

import os
import time
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.decomposition import PCA
from tensorflow.keras.datasets.mnist import load_data
from model import Triplet_Net
from utils import generate_and_save_images
from utils import plot_line
from utils import generate_and_save_3d_images
from utils import plot2dgif, plot3dgif

from tqdm import tqdm
from args import parser

opts = parser()
pd.options.mode.chained_assignment = None
'''
-----------------------Data Set-----------------------
'''
[(train_x, train_y), (test_x, test_y)] = load_data('mnist.npz')
test_x = test_x.reshape(test_x.shape[0], 28, 28, 1).astype('float32')
test_x = (test_x - 127.5) / 127.5

train_images = train_x.reshape(train_x.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_y))
train_dataset = train_dataset.shuffle(opts.BUFFER_SIZE)
train_dataset = train_dataset.batch(opts.BATCH_SIZE).take(20)
'''
-----------------------Network Setting-----------------------
'''
# method = 'batch_hard'
method = 'semi_hard'

triplet_net = Triplet_Net()
# TripletLoss = tfa.losses.TripletHardLoss(margin=opts.margin)
TripletLoss = tfa.losses.TripletSemiHardLoss(margin=opts.margin)
optimizer = tf.keras.optimizers.Adam(opts.lr, opts.beta)
'''
-----------------------Initial Log File-----------------------
'''
log_path = os.path.join(opts.LOG_PATH, method)
if not os.path.exists(log_path):
    os.mkdir(log_path)
csv_path = os.path.join(log_path, 'records.csv')
with open(csv_path, 'w') as f:
    f.write('epoch,norm,triplet_loss\n')
    
format_str = '{:5d},{:.6f},{:.6f}\n'
norm = tf.keras.metrics.Mean()
loss = tf.keras.metrics.Mean()

norm.update_state(tf.norm(triplet_net.emb_layer(test_x, training=False), axis=1))
loss.update_state(0)
# Record Loss
with open(csv_path, 'a') as f:
    f.write(format_str.format(0, norm.result().numpy(), loss.result().numpy()))
norm.reset_states()
loss.reset_states()
'''
-----------------------PCA Setting-----------------------
'''
pca = PCA(n_components=2)
pca_vec = pca.fit_transform(triplet_net(test_x, training=False))

generate_and_save_images(0, method,
                         triplet_net,
                         test_x, test_y,
                         pca)
'''
-----------------------Each Epoch Training-----------------------
'''
@tf.function
def train_step(images, label):
    with tf.GradientTape() as tape:
        trip_emb = triplet_net(images, training=True)
        triplet_loss = TripletLoss(label, trip_emb)
        
    loss.update_state(triplet_loss)
    triplet_loss = tape.gradient(triplet_loss, triplet_net.trainable_variables)
    
    optimizer.apply_gradients(zip(triplet_loss, triplet_net.trainable_variables))
    norm.update_state(tf.norm(triplet_net.emb_layer(images, training=False), axis=1))
'''
-----------------------Training-----------------------
'''
def train(train_dataset):
    # Training
    for epoch in range(opts.epochs):
        start = time.time()
        
        for image_batch, label_batch in train_dataset.as_numpy_iterator():
            train_step(image_batch, label_batch)
            
        # Record Loss
        with open(csv_path, 'a') as f:
            f.write(format_str.format(epoch+1,
                                      norm.result().numpy(),
                                      loss.result().numpy()))
        print(format_str.format(epoch+1,
                                norm.result().numpy(),
                                loss.result().numpy()))
        norm.reset_states()
        loss.reset_states()
        # Each Epoch Save Image
        generate_and_save_images(epoch + 1, method,
                                 triplet_net,
                                 test_x, test_y,
                                 pca)
        
        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            model_path = os.path.join(opts.MODEL_PATH, method)
            if not os.path.exists(model_path):
                os.mkdir(model_path)
            trip_save_path = os.path.join(model_path, 'triplet')
            triplet_net.save_weights(trip_save_path)
            
        print ('Time for epoch {} is {:.3f} sec'.format(epoch + 1, time.time()-start))
        time.sleep(0.2)

train(train_dataset)
df = pd.read_csv(csv_path)
df['triplet_loss'].iloc[0] = None
plot_line(df, 'norm', method)
plot_line(df, 'triplet_loss', method)

for view in range(0, 360, 3):
    generate_and_save_3d_images(triplet_net, test_x, test_y, view, method)

plot2dgif(method)
plot3dgif(method)
