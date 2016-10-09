#!/bin/bash
python CIFAR10_chainer.py 50  --num_fully_conn 2 --nums_conv_sets 2,2,2 --data_augmentation 2 >> results_final.txt
