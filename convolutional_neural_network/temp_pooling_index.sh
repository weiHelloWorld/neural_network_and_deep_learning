#!/bin/bash

python CIFAR10_chainer.py 30  --pooling_index "[]" >> results_pooling_index.txt
python CIFAR10_chainer.py 30  --pooling_index 1 >> results_pooling_index.txt
python CIFAR10_chainer.py 30  --pooling_index 1,2 >> results_pooling_index.txt
python CIFAR10_chainer.py 30  --pooling_index 1,2,3 >> results_pooling_index.txt
