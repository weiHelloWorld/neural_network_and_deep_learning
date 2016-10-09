#!/bin/bash

# python CIFAR10_chainer.py 30  --num_fully_conn 2 --nums_conv_sets 1,2,4 >> results_num_layers.txt
# python CIFAR10_chainer.py 30  --num_fully_conn 2 --nums_conv_sets 2,2,4 >> results_num_layers.txt
# python CIFAR10_chainer.py 30  --num_fully_conn 3 --nums_conv_sets 2,2,4 >> results_num_layers.txt
# python CIFAR10_chainer.py 30  --num_fully_conn 2 --nums_conv_sets 1,1,1 >> results_num_layers.txt
# python CIFAR10_chainer.py 30  --num_fully_conn 2 --nums_conv_sets 1,1,2 >> results_num_layers.txt
# python CIFAR10_chainer.py 30  --num_fully_conn 2 --nums_conv_sets 1,1,3 >> results_num_layers.txt
# python CIFAR10_chainer.py 30  --num_fully_conn 2 --nums_conv_sets 1,1,4 >> results_num_layers.txt
python CIFAR10_chainer.py 30  --num_fully_conn 2 --nums_conv_sets 1,2,1 >> results_num_layers.txt
python CIFAR10_chainer.py 30  --num_fully_conn 2 --nums_conv_sets 2,2,1 >> results_num_layers.txt
python CIFAR10_chainer.py 30  --num_fully_conn 2 --nums_conv_sets 2,2,2 >> results_num_layers.txt
python CIFAR10_chainer.py 30  --num_fully_conn 3 --nums_conv_sets 1,2,1 >> results_num_layers.txt
python CIFAR10_chainer.py 30  --num_fully_conn 3 --nums_conv_sets 2,2,1 >> results_num_layers.txt
python CIFAR10_chainer.py 30  --num_fully_conn 3 --nums_conv_sets 2,2,2 >> results_num_layers.txt

