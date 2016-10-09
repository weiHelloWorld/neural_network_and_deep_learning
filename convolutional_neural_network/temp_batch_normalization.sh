# shallow case
python CIFAR10_chainer.py 30 --batch_normalization 0 >> results_batch_normalization.txt
python CIFAR10_chainer.py 30 --batch_normalization 1 >> results_batch_normalization.txt
# deep case
python CIFAR10_chainer.py 30 --batch_normalization 0 --num_fully_conn 3 --nums_conv_sets 2,2,2 >> results_batch_normalization.txt
python CIFAR10_chainer.py 30 --batch_normalization 1 --num_fully_conn 3 --nums_conv_sets 2,2,2 >> results_batch_normalization.txt
