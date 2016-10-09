#!/bin/bash
python CIFAR10_chainer.py 30  --filter_size 3 --padding 1 >> results_filter_size.txt
python CIFAR10_chainer.py 30  --filter_size 5 --padding 2 >> results_filter_size.txt
python CIFAR10_chainer.py 30  --filter_size 7 --padding 3 >> results_filter_size.txt

