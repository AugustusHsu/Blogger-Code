# -*- encoding: utf-8 -*-
'''
@File    :   to_tfrecord.py
@Time    :   2021/12/09 20:02:51
@Author  :   AugustusHsu
@Contact :   jimhsu11@gmail.com
'''

from tfrecord.CelebA import CelebA2TFRecord
from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS
flags.DEFINE_string('DatasetPath', 'D:\Dataset\CelebA', 
                    'path to dataset')
flags.DEFINE_enum('Partition', 'train', ['train', 'val', 'test'], 
                  'specify train, val or test part')
flags.DEFINE_string('TFRecordPath', 
                    'D:\Blogger-Code\GAN-03 Deep Convolutional Generative Adversarial Network\data', 
                    'path to save TFRecord file')

def main(argv):
    logging.info('Covert CelebA data to TRRecord.')
    CelebA2TFRecord()
    
if __name__ == '__main__':
    app.run(main)
