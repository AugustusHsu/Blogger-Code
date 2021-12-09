# -*- encoding: utf-8 -*-
'''
@File    :   CelebA.py
@Time    :   2021/12/08 02:46:21
@Author  :   AugustusHsu
@Contact :   jimhsu11@gmail.com
'''

from absl import flags
import tensorflow as tf

FLAGS = flags.FLAGS

IMAGE_FEATURE_MAP = {
    # LandmarksTitle: ['lefteye_x',..., 'rightmouth_y']
    # AttrTitle: ['5_o_Clock_Shadow',..., 'Young']
    # bboxTitle : ['image_id', 'x_1', 'y_1', 'width', 'height']
    'image/img_celeba/filename': tf.io.FixedLenFeature([], tf.string),
    'image/img_align_celeba/filename': tf.io.FixedLenFeature([], tf.string),
    'image/img_align_celeba_png/filename': tf.io.FixedLenFeature([], tf.string),
    'image/img_celeba/encoded': tf.io.FixedLenFeature([], tf.string),
    'image/img_align_celeba/encoded': tf.io.FixedLenFeature([], tf.string),
    'image/img_align_celeba_png/encoded': tf.io.FixedLenFeature([], tf.string),
    'image/ID': tf.io.FixedLenFeature([1], tf.int64),
    'image/Attr': tf.io.FixedLenFeature([40], tf.float32),
    'image/Landmarks': tf.io.FixedLenFeature([10], tf.float32),
    'image/LandmarksAlign': tf.io.FixedLenFeature([10], tf.float32),
    'image/img_celeba/bbox': tf.io.FixedLenFeature([4], tf.float32),
}

def _load_resize_image(TFR_data, size):
    # TFR_data = tf.io.parse_single_example(tfrecord, IMAGE_FEATURE_MAP)
    raw_image = tf.image.decode_jpeg(TFR_data['image/{}/encoded'.format(FLAGS.ImageType)], channels=3)
    resize_img = tf.image.resize(raw_image, [size, size])
    resize_img = (resize_img - 127.5) / 127.5
    return resize_img

def _crop_img(TFR_data, size):
    # TFR_data = tf.io.parse_single_example(tfrecord, IMAGE_FEATURE_MAP)
    x_train = tf.image.decode_jpeg(TFR_data['image/img_celeba/encoded'], channels=3)
    
    bbox = tf.cast(TFR_data['image/img_celeba/bbox'], tf.int32)
    x, y, w, h = tf.unstack(bbox)
    
    crop_img = tf.image.crop_to_bounding_box(x_train, y, x, h, w)
    resize_img = tf.image.resize(crop_img, [size, size])
    resize_img = (resize_img - 127.5) / 127.5
    return resize_img

def _get_landmarks(TFR_data):
    if FLAGS.ImageType == 'img_celeba':
        Landmarks = TFR_data['image/Landmarks']
    else:
        Landmarks = TFR_data['image/LandmarksAlign']
    
    # Landmarks = tf.unstack(Landmarks)
    # lefteye_x, lefteye_y, \
    #     righteye_x, righteye_y, \
    #         nose_x, nose_y, \
    #             leftmouth_x, leftmouth_y, \
    #                 rightmouth_x, rightmouth_y = Landmarks
    # return (lefteye_x, lefteye_y), (righteye_x, righteye_y), \
    #     (nose_x, nose_y), (leftmouth_x, leftmouth_y), (rightmouth_x, rightmouth_y)
    return Landmarks

def _get_attr(TFR_data):
    return TFR_data['image/Attr']

@tf.function
def _your_dataset(tfrecord, size):
    TFR_data = tf.io.parse_single_example(tfrecord, IMAGE_FEATURE_MAP)
    if FLAGS.ImageType == 'img_celeba' and FLAGS.bbox:
        resize_img = _crop_img(TFR_data, size)
    else:
        resize_img = _load_resize_image(TFR_data, size)
    Landmarks = _get_landmarks(TFR_data)
    Attr = _get_attr(TFR_data)
    return resize_img, Landmarks, Attr

def LoadTFRecordDataset(TFRecordPath):
    dataset = tf.data.Dataset.list_files(TFRecordPath)
    dataset = dataset.flat_map(tf.data.TFRecordDataset)
    return dataset.map(lambda x: _your_dataset(x, FLAGS.DisInSize), num_parallel_calls=-1)