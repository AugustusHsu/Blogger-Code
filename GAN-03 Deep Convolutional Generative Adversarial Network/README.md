# GAN-03 Deep Convolutional Generative Adversarial Network

透過DCGAN來生成MNIST、CelebA的圖片。

## CelebA

CelebA to tfrecord test and tfrecord to tf.dataset test

```bash
test_tfrecord.py:
  --DatasetPath: path to dataset
    (default: 'D:\\Dataset\\CelebA')
  --Partition: <train|val|test>: specify train, val or test part
    (default: 'train')
  --TFRecordPath: path to save TFRecord file
    (default: 'D:\\Blogger-Code\\GAN-03 Deep Convolutional Generative
    Adversarial Network\\data')

test_dataset.py:
  --DatasetPath: path to dataset
    (default: 'YourDatasetPath')
  --DisInSize: discriminator input image size
    (default: '64')
    (an integer)
  --ImageType: <img_celeba|img_align_celeba|img_align_celeba_png>: specify
    train, val or test part
    (default: 'img_celeba')
  --Partition: <train|val|test>: specify train, val or test part
    (default: 'test')
  --TFRecordPath: path to save TFRecord file
    (default: 'TFRecordPath')
  --[no]bbox: crop image by bbox(only run when ImageType is img_celeba)
    (default: 'true')
```

