python test_tfrecord.py --Partition train
python test_tfrecord.py --Partition val
python test_tfrecord.py --Partition test

python test_dataset.py --Partition train
python test_dataset.py --Partition val
python test_dataset.py --Partition test
python test_dataset.py --Partition train --ImageType img_align_celeba
python test_dataset.py --Partition val --ImageType img_align_celeba
python test_dataset.py --Partition test --ImageType img_align_celeba
python test_dataset.py --Partition train --ImageType img_align_celeba_png
python test_dataset.py --Partition val --ImageType img_align_celeba_png
python test_dataset.py --Partition test --ImageType img_align_celeba_png