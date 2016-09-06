from feedforward_full_connection_network import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("node_num_of_hidden_layer", type=int)
parser.add_argument("hidden_layer_type", type=str)
parser.add_argument("learning_rate", type=float)
parser.add_argument("num_of_epochs", type=int)
parser.add_argument("batch_size", type=int)
parser.add_argument("cost_func_der", type=str)
args = parser.parse_args()

# Load the dataset
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

train_input = train_set[0]
num_of_data = train_input.shape[0]
train_output = np.zeros((num_of_data, 10))
for item in range(num_of_data):
    train_output[item][train_set[1][item]] = 1

if args.hidden_layer_type == "tanh":
    hidden_layer = tanh_layer()
elif args.hidden_layer_type == "sigmoid":
    hidden_layer = sigmoid_layer()
elif args.hidden_layer_type == "relu":
    hidden_layer = relu_layer()
else:
    raise Exception('wrong hidden_layer_type')

if args.cost_func_der == "mse":
    cost_function_der = mse_der
elif args.cost_func_der == "log_likelihood":
    cost_function_der = log_likelihood_der
else:
    raise Exception('wrong cost_func_der')

a = network(learning_rate=args.learning_rate, num_of_nodes=[784, args.node_num_of_hidden_layer, 10], 
            input_data = train_input, output_data=train_output, 
            cost_func_der = cost_function_der,
            layer_list=[hidden_layer, softmax_layer()])

for item in range(args.num_of_epochs):
    a.train_one_epoch(batch_size=args.batch_size)
    result = np.array([item.argmax() for item in a.predict(test_set[0])])
    print sum((result - test_set[1]) != 0) / float(len(test_set[1]))

print ("Done")
