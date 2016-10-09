#!/bin/bash
python CIFAR10_chainer.py 100  --num_fully_conn 2 --nums_conv_sets 1,1,1 --data_augmentation 0 >> results_data_augmentation.txt
python CIFAR10_chainer.py 50  --num_fully_conn 2 --nums_conv_sets 1,1,1 --data_augmentation 1 >> results_data_augmentation.txt

