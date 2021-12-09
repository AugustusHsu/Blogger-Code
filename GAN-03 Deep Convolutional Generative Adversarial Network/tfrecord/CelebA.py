# -*- encoding: utf-8 -*-
'''
@File    :   CelebA.py
@Time    :   2021/12/08 02:44:18
@Author  :   AugustusHsu
@Contact :   jimhsu11@gmail.com
'''

import os
from tqdm import tqdm
from time import sleep
from absl import flags
from absl import logging
import tensorflow as tf

FLAGS = flags.FLAGS

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
def float_list_feature(value):
     return tf.train.Feature(float_list=tf.train.FloatList(value=value))
def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def load_txt(path, skip_rows=0):
    with open(path, 'r') as f:
        my_dict = {}
        data = f.readlines()
        if skip_rows != 0:
            title = data[1]
            title = title.strip()
            title = title.split(' ')
            title = [x for x in title if x != '']
            data = data[skip_rows:]
        else:
            title = None
        for line in data:
            line = line.strip()
            line = line.split(' ')
            line = [x for x in line if x != '']
            my_dict[line[0]] = line[1:]
        return my_dict, title

def build_example(Data, key):
    PartitionData, IdentityData, AttrData, LandmarksData, LandmarksAlignData, bboxData = Data
    ImagePath = os.path.join(FLAGS.DatasetPath, 'Img', 'img_celeba', key)
    AlignImagePath = os.path.join(FLAGS.DatasetPath, 'Img', 'img_align_celeba', key)
    AlignPNGPath = os.path.join(FLAGS.DatasetPath, 'Img', 'img_align_celeba_png',  key[:-4] + '.png')
    ImageRaw = open(ImagePath, 'rb').read()
    AlignImageRaw = open(AlignImagePath, 'rb').read()
    AlignPNGRaw = open(AlignPNGPath, 'rb').read()
    
    ID = int(IdentityData[key][0])
    Attr = [0.0 if int(attr) < 0 else 1.0 for attr in AttrData[key]]
    Landmarks = [float(landmark) for landmark in LandmarksData[key]]
    LandmarksAlign = [float(landmark) for landmark in LandmarksAlignData[key]]
    bbox = [float(x) for x in bboxData[key]]
    
    png_key = key[:-4] + '.png'
    example = tf.train.Example(features=tf.train.Features(feature={
      'image/img_celeba/filename': bytes_feature(key.encode('utf8')),
      'image/img_align_celeba/filename': bytes_feature(key.encode('utf8')),
      'image/img_align_celeba_png/filename': bytes_feature(png_key.encode('utf8')),
      'image/img_celeba/encoded': bytes_feature(ImageRaw),
      'image/img_align_celeba/encoded': bytes_feature(AlignImageRaw),
      'image/img_align_celeba_png/encoded': bytes_feature(AlignPNGRaw),
      'image/ID': int64_feature(ID),
      'image/Attr': float_list_feature(Attr),
      'image/Landmarks': float_list_feature(Landmarks),
      'image/LandmarksAlign': float_list_feature(LandmarksAlign),
      'image/img_celeba/bbox': float_list_feature(bbox),
    }))
    return example

def CelebA2TFRecord():
    PartitionPath = os.path.join(FLAGS.DatasetPath, 'Eval', 'list_eval_partition.txt')
    Anno = os.path.join(FLAGS.DatasetPath, 'Anno')
    IdentityPath = os.path.join(Anno, 'identity_CelebA.txt')
    AttrPath = os.path.join(Anno, 'list_attr_celeba.txt')
    bboxPath = os.path.join(Anno, 'list_bbox_celeba.txt')
    LandmarksPath = os.path.join(Anno, 'list_landmarks_celeba.txt')
    LandmarksAlignPath = os.path.join(Anno, 'list_landmarks_align_celeba.txt')
    
    logging.info('Load {} part Index File'.format(FLAGS.Partition))
    PartitionData, _ = load_txt(PartitionPath, 0)
    
    logging.info('Load text data ')
    IdentityData, _ = load_txt(IdentityPath, 0)
    AttrData, AttrTitle = load_txt(AttrPath, 2)
    bboxData, bboxTitle = load_txt(bboxPath, 2)
    LandmarksData, LandmarksTitle = load_txt(LandmarksPath, 2)
    LandmarksAlignData, LandmarksAlignTitle = load_txt(LandmarksAlignPath, 2)
    
    logging.info('Create TFRecord folder')
    os.makedirs(FLAGS.TFRecordPath, exist_ok=True)
    TFRecordFile = os.path.join(FLAGS.TFRecordPath, 'CelebA' + '_{}.tfrecord').format(FLAGS.Partition)
    logging.info(TFRecordFile)
    logging.info('Create TFRecord writer')
    writer = tf.io.TFRecordWriter(TFRecordFile)
    
    Data = PartitionData, IdentityData, AttrData, LandmarksData, LandmarksAlignData, bboxData
    Title = AttrTitle, LandmarksTitle, LandmarksAlignTitle, bboxTitle
        
    part_dict = {'train':'0', 'val':'1', 'test':'2'}
    for key in tqdm(PartitionData.keys()):
        # 101283.jpg的bbox w, h 為0
        if key == '101283.jpg':
            continue
        if PartitionData[key][0] == part_dict[FLAGS.Partition]:
            tf_example = build_example(Data, key)
            writer.write(tf_example.SerializeToString())
    writer.close()
    sleep(0.3)
    logging.info("Done")
