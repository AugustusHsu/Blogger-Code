# -*- encoding: utf-8 -*-
'''
@File    :   run.py
@Time    :   2021/12/09 20:06:23
@Author  :   AugustusHsu
@Contact :   jimhsu11@gmail.com
'''

import os
from tqdm import tqdm
from dataset.CelebA import LoadTFRecordDataset
from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS
flags.DEFINE_string('DatasetPath', 'D:\Dataset\CelebA', 
                    'path to dataset')
flags.DEFINE_enum('Partition', 'test', ['train', 'val', 'test'], 
                  'specify train, val or test part')
flags.DEFINE_enum('ImageType', 'img_celeba', ['img_celeba', 'img_align_celeba', 'img_align_celeba_png'], 
                  'specify train, val or test part')
flags.DEFINE_boolean('bbox', True, 'crop image by bbox(only run when ImageType is img_celeba)')
flags.DEFINE_string('TFRecordPath', 
                    'D:\Blogger-Code\GAN-03 Deep Convolutional Generative Adversarial Network\data', 
                    'path to save TFRecord file')
flags.DEFINE_integer('DisInSize', 64, 'discriminator input image size')

def main(argv):
    logging.info('Read TFRecord as tf.dataset.')
    TFRecordPath = os.path.join(FLAGS.TFRecordPath, 'CelebA_{}.tfrecord'.format(FLAGS.Partition))
    logging.info(TFRecordPath)
    dataset = LoadTFRecordDataset(TFRecordPath)
    for data, data2, data3 in tqdm(dataset):
        pass
    
if __name__ == '__main__':
    app.run(main)
