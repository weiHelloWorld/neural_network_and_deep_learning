#!/bin/bash
python CIFAR10_chainer.py 30  --num_fully_conn 3 --nums_conv_sets 2,2,2 --data_augmentation 2 --optimizer Adam >> results_final.txt
