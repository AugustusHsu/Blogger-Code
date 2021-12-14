# GAN-03 Deep Convolutional Generative Adversarial Network

透過DCGAN來生成MNIST、CelebA的圖片。

## CelebA Dataset

CelebA to tfrecord test and tfrecord to tf.dataset test

```bash
test_tfrecord.py:
  --DatasetPath: path to dataset
    (default: 'YourDatasetPath')
  --Partition: <train|val|test>: specify train, val or test part
    (default: 'train')
  --TFRecordPath: path to save TFRecord file
    (default: 'TFRecordFilePath')

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

## Argument of run.py

```bash
run.py:
  --BATCH_SIZE: batch size
    (default: '256')
    (an integer)
  --DatasetPath: path to dataset
    (default: 'YourDatasetPath')
  --DisInSize: discriminator input image size
    (default: '64')
    (an integer)
  --ImageType: <img_celeba|img_align_celeba|img_align_celeba_png|MNIST>: specify
    celeba image type or use mnist
    (default: 'img_celeba')
  --LOG_PATH: path to log_dir
    (default: 'logs/')
  --MODEL_PATH: path to save the model
    (default: 'models/')
  --Partition: <train|val|test|all>: specify train, val or test part
    (default: 'all')
  --TFRecordPath: path to save TFRecord file
    (default: 'TFRecordPath')
  --[no]bbox: crop image by bbox(only run when ImageType is img_celeba)
    (default: 'true')
  --epochs: epoch
    (default: '35')
    (an integer)
  --lr: learning rate
    (default: '0.0002')
    (a number)
  --noise_dim: noise dimension
    (default: '100')
    (an integer)
```

