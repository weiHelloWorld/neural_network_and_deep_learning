"""This file is modified from the code provided by the instructor
The model is the VGG model from https://github.com/mitmul/chainer-cifar10
The original paper is "Very Deep Convolutional Networks for Large-Scale Image Recognition"
"""
import numpy as np
import time, argparse, os, h5py
from datetime import datetime
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

parser = argparse.ArgumentParser()
parser.add_argument("num_epochs", type=int)
parser.add_argument("--nums_conv_sets", type=str, default = "1,1,1")
parser.add_argument("--num_fully_conn", type=int, default = 2)
parser.add_argument("--batch_normalization", type=int, default = 1)
parser.add_argument("--dropout_prop", type=float, default=0.25)
parser.add_argument("--filter_size", type=int, default=3)
parser.add_argument("--padding", type=int, default=1)
parser.add_argument("--pooling_index",type=str, default="[1,2,3]")   # index of pooling layers added
parser.add_argument("--optimizer", type=str, default="MomentumSGD")
parser.add_argument("--pooling_type", type=str, default="max")
parser.add_argument("--data_augmentation", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=200)
args = parser.parse_args()

# print all arguments
print "################################"
print "arguments are:"
for arg in vars(args):
    print arg, getattr(args, arg)
print "################################"

# load CIFAR10 data
CIFAR10_data = h5py.File('CIFAR10.hdf5', 'r')
x_train = np.float32(CIFAR10_data['X_train'][:] )
y_train = np.int32(np.array(CIFAR10_data['Y_train'][:]))
x_test = np.float32(CIFAR10_data['X_test'][:] )
y_test = np.int32( np.array(CIFAR10_data['Y_test'][:]) )

CIFAR10_data.close()

if args.data_augmentation == 1:
    x_train = np.concatenate((x_train, x_train[:,:,:,::-1]), axis=0)
    y_train = np.concatenate((y_train, y_train), axis=0)
elif args.data_augmentation == 2:
    x_train = np.concatenate((x_train, x_train[:,:,:,::-1], x_train[:,:,::-1,:], x_train[:,:,::-1,::-1]), axis=0)
    y_train = np.concatenate((y_train, y_train, y_train, y_train), axis=0)

print("data loaded, number of training data = %d" % x_train.shape[0])

cuda.cudnn_enabled = True
nums_conv_sets = [int(item) for item in args.nums_conv_sets.strip().split(",")]
if args.pooling_type == "max":
    pooling_type = F.max_pooling_2d
elif args.pooling_type == 'average':
    pooling_type = F.average_pooling_2d
else:
    raise Exception('wrong pooling type')

pooling_index = args.pooling_index.replace('[','').replace(']','')

class Conv_NN(Chain):
    def __init__(self):
        super(Conv_NN, self).__init__(
            conv1_1=L.Convolution2D(3, 64, args.filter_size, pad=args.padding),
            bn1_1=L.BatchNormalization(64),
            conv1_2=L.Convolution2D(64, 64, args.filter_size, pad=args.padding),
            bn1_2=L.BatchNormalization(64),

            conv2_1=L.Convolution2D(64, 128, args.filter_size, pad=args.padding),
            bn2_1=L.BatchNormalization(128),
            conv2_2=L.Convolution2D(128, 128, args.filter_size, pad=args.padding),
            bn2_2=L.BatchNormalization(128),

            conv3_1=L.Convolution2D(128, 256, args.filter_size, pad=args.padding),
            bn3_1=L.BatchNormalization(256),
            conv3_2=L.Convolution2D(256, 256, args.filter_size, pad=args.padding),
            bn3_2=L.BatchNormalization(256),

            fc4 = L.Linear(4096 * 64 / (4 ** int((len(pooling_index) + 1) / 2) ), 500),
            fc5 = L.Linear(500, 500),                                         
            fc6 = L.Linear(500,10),
        )
    def __call__(self, x_data, y_data, dropout_bool, bn_bool, p):
        x = Variable(x_data)
        t = Variable(y_data)
        h = F.relu(self.bn1_1(self.conv1_1(x), bn_bool)) if args.batch_normalization else F.relu(self.conv1_1(x))
        for item in range(nums_conv_sets[0] - 1):
            h = F.relu(self.bn1_2(self.conv1_2(h), bn_bool)) if args.batch_normalization else F.relu(self.conv1_2(h))

        if "1" in pooling_index:
            h = pooling_type(h, 2, 2)

        h = F.dropout(h, p, dropout_bool)

        h = F.relu(self.bn2_1(self.conv2_1(h), bn_bool)) if args.batch_normalization else F.relu(self.conv2_1(h))
        for item in range(nums_conv_sets[1] - 1):
            h = F.relu(self.bn2_2(self.conv2_2(h), bn_bool)) if args.batch_normalization else F.relu(self.conv2_2(h))

        if "2" in pooling_index:
            h = pooling_type(h, 2, 2)

        h = F.dropout(h, p, dropout_bool)

        h = F.relu(self.bn3_1(self.conv3_1(h), bn_bool)) if args.batch_normalization else F.relu(self.conv3_1(h))
        for item in range(nums_conv_sets[2] - 1):
            h = F.relu(self.bn3_2(self.conv3_2(h), bn_bool)) if args.batch_normalization else F.relu(self.conv3_2(h))
        
        if "3" in pooling_index:
            h = pooling_type(h, 2, 2)

        h = F.dropout(h, p, dropout_bool)

        h = F.dropout(F.relu(self.fc4(h)), p, dropout_bool)
        for item in range(args.num_fully_conn - 2):
            h = F.dropout(F.relu(self.fc5(h)), p, dropout_bool)

        L_out = self.fc6(h)
        return F.softmax_cross_entropy(L_out, t), F.accuracy(L_out, t)

#returns test accuracy of the model.  dropout is set to its test state 
def Calculate_Test_Accuracy(x_test, y_test, model, p, GPU_on, batch_size):
    L_Y_test = len(y_test)
    counter = 0
    test_accuracy_total = 0.0
    for i in range(0, L_Y_test, batch_size):
        if (GPU_on):
            x_batch = cuda.to_gpu(x_test[i:i+ batch_size,:])
            y_batch = cuda.to_gpu(y_test[i:i+ batch_size] )
        else:
            x_batch = x_test[i:i+batch_size,:]
            y_batch = y_test[i:i+batch_size]
        dropout_bool = False
        bn_bool = True
        loss, accuracy = model(x_batch, y_batch, dropout_bool,bn_bool, p)
        test_accuracy_batch  = 100.0*np.float(accuracy.data )
        test_accuracy_total += test_accuracy_batch
        counter += 1
    test_accuracy = test_accuracy_total/(np.float(counter))
    return test_accuracy


model =  Conv_NN()

#True if training with GPU, False if training with CPU
GPU_on = True

#size of minibatches
batch_size = args.batch_size
test_batch_size = 10

#transfer model to GPU
if (GPU_on):
    model.to_gpu()

#optimization method
if args.optimizer == 'MomentumSGD':
    optimizer = optimizers.MomentumSGD(momentum = .99)
    optimizer.lr = .01
elif args.optimizer == "SGD":
    optimizer = optimizers.MomentumSGD(momentum = 0)
    optimizer.lr = .01
elif args.optimizer == 'RMSprop':
    optimizer = optimizers.RMSprop(lr=0.001, alpha=0.99, eps=1e-08)
elif args.optimizer == 'AdaGrad':
    optimizer = optimizers.AdaGrad()
elif args.optimizer == 'Adam':
    optimizer = optimizers.Adam()
else:
    raise Exception('wrong optimizer')

optimizer.setup(model)

#dropout probability
dropout_prop = args.dropout_prop

#number of training epochs
num_epochs = args.num_epochs

L_Y_train = len(y_train)

time1 = time.time()
max_train_accuracy = 0
max_train_accuracy_epoch = 0

for epoch in range(num_epochs):
    #reshuffle dataset
    I_permutation = np.random.permutation(L_Y_train)
    x_train = x_train[I_permutation,:]
    y_train = y_train[I_permutation]
    epoch_accuracy = 0.0
    batch_counter = 0
    for i in range(0, L_Y_train, batch_size):
        if (GPU_on):
            x_batch = cuda.to_gpu(x_train[i:i+batch_size,:])
            y_batch = cuda.to_gpu(y_train[i:i+batch_size] )
        else:
            x_batch = x_train[i:i+batch_size,:]
            y_batch = y_train[i:i+batch_size]     
        model.zerograds()
        dropout_bool = True
        bn_bool = False
        loss, accuracy = model(x_batch, y_batch, dropout_bool, bn_bool, dropout_prop)
        loss.backward()
        optimizer.update()
        #print("success")
        epoch_accuracy += np.float(accuracy.data)
        batch_counter += 1
    
    train_accuracy = 100.0*epoch_accuracy/np.float(batch_counter) 
    print "Train accuracy: %f" % train_accuracy
    if max_train_accuracy < train_accuracy:
        max_train_accuracy_epoch = epoch
        max_train_accuracy = train_accuracy
    if epoch == num_epochs - 1 or max_train_accuracy_epoch < epoch - 5:  # a simplified version of early stopping based on training accuracy
        test_accuracy = Calculate_Test_Accuracy(x_test, y_test, model, dropout_prop, GPU_on, test_batch_size)
        print "Test Accuracy: %f" % test_accuracy
        break

    
time2 = time.time()
training_time = time2 - time1

try:
    print "Rank: %d" % rank
except Exception as e:
    pass
    
print "Training time: %f" % training_time
print "Training time per epoch: %f" % (training_time / (epoch + 1))
