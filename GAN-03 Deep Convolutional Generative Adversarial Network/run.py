# -*- encoding: utf-8 -*-
'''
@File    :   run.py
@Time    :   2021/12/10 03:00:50
@Author  :   AugustusHsu
@Contact :   jimhsu11@gmail.com
'''

import os
import time
import pandas as pd
import tensorflow as tf
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from model import Dis_Net, Gen_Net
from utils import generator_loss, discriminator_loss
from utils import generate_and_save_images
from utils import plot_GIF, plot_line
from tqdm import tqdm
from absl import app
from absl import flags
from absl import logging
from dataset.CelebA import LoadTFRecordDataset

FLAGS = flags.FLAGS
flags.DEFINE_string('DatasetPath', 'D:\Dataset\CelebA', 
                    'path to dataset')
flags.DEFINE_enum('Partition', 'all', ['train', 'val', 'test', 'all'], 
                  'specify train, val or test part')
flags.DEFINE_enum('ImageType', 'img_celeba', ['img_celeba', 'img_align_celeba', 'img_align_celeba_png', 'MNIST'], 
                  'specify celeba image type or use mnist')
flags.DEFINE_boolean('bbox', True, 'crop image by bbox(only run when ImageType is img_celeba)')

flags.DEFINE_string('TFRecordPath', 
                    'D:\Blogger-Code\GAN-03 Deep Convolutional Generative Adversarial Network\data', 
                    'path to save TFRecord file')
flags.DEFINE_string('LOG_PATH', 'logs/', 'path to log_dir')
flags.DEFINE_string('MODEL_PATH', 'models/', 'path to save the model')

flags.DEFINE_integer('DisInSize', 64, 'discriminator input image size')
flags.DEFINE_integer('noise_dim', 100, 'noise dimension')
flags.DEFINE_integer('BATCH_SIZE', 256, 'batch size')
flags.DEFINE_float('lr', 2e-4, 'learning rate')
flags.DEFINE_integer('epochs', 35, 'epoch')

def create_dataset():
    '''
    -----------------------Data Set-----------------------
    '''
    if  FLAGS.ImageType == 'MNIST':
        FLAGS.bbox = False
        [(train_x, train_y), (test_x, test_y)] = load_data('mnist.npz')

        train_images = train_x.reshape(train_x.shape[0], 28, 28, 1).astype('float32')
        train_images = (train_images - 127.5) / 127.5

        train_dataset = tf.data.Dataset.from_tensor_slices(train_images)
        train_dataset = train_dataset.map(lambda x: tf.image.resize_with_pad(x, FLAGS.DisInSize, FLAGS.DisInSize))
    else:
        logging.info('Covert CelebA data to TRRecord.')
        if FLAGS.Partition == 'all':
            TFRecordPath = os.path.join(FLAGS.TFRecordPath, 'CelebA_train.tfrecord')
            logging.info(TFRecordPath)
            train_dataset = LoadTFRecordDataset(TFRecordPath)
            for partition in ['val', 'test']:
                TFRecordPath = os.path.join(FLAGS.TFRecordPath, 'CelebA_{}.tfrecord'.format(partition))
                logging.info(TFRecordPath)
                part_dataset = LoadTFRecordDataset(TFRecordPath)
                train_dataset.concatenate(part_dataset)
        else:
            TFRecordPath = os.path.join(FLAGS.TFRecordPath, 'CelebA_{}.tfrecord'.format(FLAGS.Partition))
            logging.info(TFRecordPath)
            train_dataset = LoadTFRecordDataset(TFRecordPath)
    train_dataset = train_dataset.shuffle(300)
    train_dataset = train_dataset.batch(FLAGS.BATCH_SIZE)
    train_dataset = train_dataset.prefetch(20)
        
    NOISE = tf.random.normal([16, FLAGS.noise_dim])
    return train_dataset, NOISE

def setup_model():
    if FLAGS.ImageType == 'MNIST':
        Generator = Gen_Net(1)
        Discriminator = Dis_Net([FLAGS.DisInSize, FLAGS.DisInSize], 1)
    else:
        Generator = Gen_Net(3)
        Discriminator = Dis_Net([FLAGS.DisInSize, FLAGS.DisInSize], 3)
    if FLAGS.ImageType == 'MNIST':
        one_epoch_size = 60000//FLAGS.BATCH_SIZE + 1
    else:
        one_epoch_size = 162769//FLAGS.BATCH_SIZE + 1
    boundaries = [one_epoch_size*15, one_epoch_size*25, one_epoch_size*28]
    values = [FLAGS.lr, FLAGS.lr/10, FLAGS.lr/100, FLAGS.lr/1000]
    lr_fn = PiecewiseConstantDecay(boundaries, values)

    G_opt = tf.keras.optimizers.Adam(lr_fn, 0.5)
    D_opt = tf.keras.optimizers.Adam(lr_fn, 0.5)
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
    -----------------------Initial-----------------------
    '''
    SaveName = '{}-{}'.format(FLAGS.ImageType, str(FLAGS.bbox))
    train_dataset, NOISE = create_dataset()
    # Initial Log File
    if not os.path.exists(FLAGS.LOG_PATH):
        os.mkdir(FLAGS.LOG_PATH)
    csv_path = os.path.join(FLAGS.LOG_PATH, '{}-loss.csv'.format(SaveName))
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
        generate_and_save_images(Generator((NOISE), training=False), epoch + 1, SaveName)
        
        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            Gen_save_path = os.path.join(FLAGS.MODEL_PATH, SaveName, 'Generator')
            Dis_save_path = os.path.join(FLAGS.MODEL_PATH, SaveName, 'Discriminator')
            Generator.save_weights(Gen_save_path)
            Discriminator.save_weights(Dis_save_path)
        
        logging.info('Time for epoch {} is {:.3f} sec'.format(epoch + 1, time.time()-start))
        time.sleep(0.2)
    
    plot_GIF(SaveName)

    df = pd.read_csv(csv_path)
    col_name = ['Real_P', 'Fake_P']
    plot_line(df, col_name, SaveName, figname='probability')
    col_name = ['Gen_loss', 'Dis_loss']
    plot_line(df, col_name, SaveName, figname='loss')

if __name__ == '__main__':
    app.run(main)
