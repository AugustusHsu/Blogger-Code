# -*- encoding: utf-8 -*-
'''
@File    :   run.py
@Time    :   2021/11/30 15:15:23
@Author  :   AugustusHsu
@Contact :   jimhsu11@gmail.com
'''

import os
import time
import pandas as pd
import tensorflow as tf
from tensorflow.keras.datasets.mnist import load_data
from model import Dis_Net, Gen_Net
from utils import generator_loss, discriminator_loss
from utils import generate_and_save_images
from utils import plot_GIF, plot_line
from tqdm import tqdm
from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS
flags.DEFINE_string('LOG_PATH', 'logs/', 
                    'path to log_dir')
flags.DEFINE_string('MODEL_PATH', 'models/', 
                    'path to save the model')
flags.DEFINE_integer('noise_dim', 100, 'noise dimension')
flags.DEFINE_integer('BATCH_SIZE', 256, 'batch size')
flags.DEFINE_float('lr', 2e-4, 'learning rate')
flags.DEFINE_integer('epochs', 120, 'epoch')

def setup_model():
    Generator = Gen_Net()
    Discriminator = Dis_Net()

    G_opt = tf.keras.optimizers.Adam(FLAGS.lr*5, 0.5)
    D_opt = tf.keras.optimizers.Adam(FLAGS.lr, 0.5)
    return Generator, Discriminator, G_opt, D_opt

@tf.function
def train_step(models, opts, images, loss):
    Generator, Discriminator = models
    G_opt, D_opt = opts
    noise = tf.random.normal([images.shape[0], FLAGS.noise_dim])
    
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

def main(argv):
    '''
    -----------------------Data Set-----------------------
    '''
    [(train_x, train_y), (test_x, test_y)] = load_data('mnist.npz')

    train_images = train_x.reshape(train_x.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5

    train_dataset = tf.data.Dataset.from_tensor_slices(train_images)
    train_dataset = train_dataset.shuffle(60000)
    train_dataset = train_dataset.batch(FLAGS.BATCH_SIZE)

    NOISE = tf.random.normal([16, FLAGS.noise_dim])
    
    '''
    -----------------------Initial-----------------------
    '''
    # Initial Log File
    log_path = os.path.join(FLAGS.LOG_PATH)
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
    
    Generator, Discriminator, G_opt, D_opt = setup_model()
    models = [Generator, Discriminator]
    opts = [G_opt, D_opt]
    
    '''
    -----------------------Training-----------------------
    '''
    for epoch in range(FLAGS.epochs):
        start = time.time()
        for image_batch in tqdm(train_dataset.as_numpy_iterator()):
            train_step(models, opts, image_batch, loss)
        
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
        generate_and_save_images(Generator(NOISE, training=False), epoch + 1)
        
        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            Gen_save_path = os.path.join(FLAGS.MODEL_PATH, 'Generator')
            Dis_save_path = os.path.join(FLAGS.MODEL_PATH, 'Discriminator')
            Generator.save_weights(Gen_save_path)
            Discriminator.save_weights(Dis_save_path)
        
        logging.info('Time for epoch {} is {:.3f} sec'.format(epoch + 1, time.time()-start))
        time.sleep(0.2)
    plot_GIF()

    csv_path = os.path.join(FLAGS.LOG_PATH, '{}.csv'.format('loss'))
    df = pd.read_csv(csv_path)
    col_name = ['Real_P', 'Fake_P']
    plot_line(df, col_name, 'gan', figname='probability')
    col_name = ['Gen_loss', 'Dis_loss']
    plot_line(df, col_name, 'gan', figname='loss')

if __name__ == '__main__':
    app.run(main)