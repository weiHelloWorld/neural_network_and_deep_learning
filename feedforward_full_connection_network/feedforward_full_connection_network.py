import cPickle, gzip, numpy as np, copy, random

def log_likelihood_der(expected, actual):
    return - expected / actual

def mse_der(expected, actual):
    return (actual - expected)

class network(object):
    def __init__(self, learning_rate, num_of_nodes, input_data, output_data, 
                 cost_func_der,     # derivation of cost function 
                 connection_list = None, 
                 layer_list=None
                 ):
        self._learning_rate = learning_rate
        self._num_of_nodes = num_of_nodes
        self._cost_func_der = cost_func_der
        self._in_data_list = list(range(len(num_of_nodes)))
        self._out_data_list = list(range(len(num_of_nodes)))
        self._in_err_list = list(range(len(num_of_nodes)))
        self._out_err_list = list(range(len(num_of_nodes)))
        self._out_data_list_list = []   # used to store the history of out_data_list
        self._in_err_list_list = []
        self._connection_list = connection_list
        if self._connection_list is None:
            self._connection_list = [full_connection(num_of_nodes[_1], num_of_nodes[_1 + 1]) 
                                     for _1 in range(len(num_of_nodes) - 1)]
        self._layer_list = layer_list
        assert (len(self._connection_list) == len(self._layer_list) == len(num_of_nodes) - 1)
        self._input_data = input_data
        self._output_data = output_data
        return
    
    def forward(self, single_input):
        last_index = len(self._num_of_nodes) - 1
        self._in_data_list[0] = self._out_data_list[0] = single_input
        for item in range(len(self._connection_list)):
            self._in_data_list[item + 1] = self._connection_list[item].forward(self._out_data_list[item])
            self._out_data_list[item + 1] = self._layer_list[item].forward(self._in_data_list[item + 1])
        return self._out_data_list[last_index]
    
    def backward(self, single_output):
        last_index = len(self._num_of_nodes) - 1
        self._out_err_list[last_index] = self._cost_func_der(single_output, self._out_data_list[last_index])
        for item in reversed(range(len(self._connection_list))):
            self._in_err_list[item + 1] = self._layer_list[item].backward(self._out_data_list[item + 1], self._out_err_list[item + 1])
            self._out_err_list[item] = self._connection_list[item].backward(self._in_err_list[item + 1])
        return
    
    def update_coeff_and_bias_and_clear_history(self):
        assert (len(self._out_data_list_list) == len(self._in_err_list_list)), (len(self._out_data_list_list), len(self._in_err_list_list))
        for temp_out_data_list, temp_in_err_list in zip(self._out_data_list_list, self._in_err_list_list):
            for item in range(len(self._connection_list)):
                self._connection_list[item].update_coeff_and_bias(temp_out_data_list[item], 
                                                                  temp_in_err_list[item + 1], self._learning_rate)
        # clear history list when done
        self._out_data_list_list = []
        self._in_err_list_list = []
        return
        
    def predict(self, input_data):
        assert (input_data.shape[1] == self._num_of_nodes[0])
        result = [self.forward(item) for item in input_data]
        return np.array(result)

    def train_one_epoch(self, batch_size=1):  # support mini-batch SGD training
        input_data = self._input_data
        output_data = self._output_data
        training_set = zip(input_data, output_data)
        random.shuffle(training_set)     # SGD
        for _1, item in enumerate(training_set):
            self.forward(item[0])
            self.backward(item[1])
            self._out_data_list_list.append(copy.deepcopy(self._out_data_list))
            self._in_err_list_list.append(copy.deepcopy(self._in_err_list))
            if (_1 + 1) % batch_size == 0:     
                self.update_coeff_and_bias_and_clear_history()

        return

    def train_n_epoches(self, num_of_epochs, batch_size = 1):
        for _ in range(num_of_epochs):
            self.train_one_epoch(batch_size)
        return
        
        
class full_connection(object):
    def __init__(self, in_dim, out_dim):
        self._coeff = np.random.rand(out_dim, in_dim) - 0.5    # initialize randomly
        self._bias = np.random.rand(out_dim) - 0.5  
        return
        
    def forward(self, in_data):
        assert (self._coeff.shape[1] == in_data.shape[0]), (self._coeff.shape[1] , in_data.shape[0])
        return np.dot(self._coeff, in_data) + self._bias
    
    def backward(self, out_err):
        return np.dot(self._coeff.T, out_err)
    
    def update_coeff_and_bias(self, in_data_of_previous_layer, out_err, learning_rate):
        self._coeff -= learning_rate * np.outer(out_err, in_data_of_previous_layer)
        self._bias -= learning_rate * out_err
        return

class tanh_layer(object):
    def forward(self, in_data):
        return np.tanh(in_data)
    
    def backward(self, out_data, out_err):
        return (1 - out_data ** 2) * out_err
    
class sigmoid_layer(object):
    def forward(self, in_data):
        return 1 / (1 + np.exp(- in_data))
    
    def backward(self, out_data, out_err):
        return out_data * (1 - out_data) * out_err

class linear_layer(object):
    def forward(self, in_data):
        return in_data
    
    def backward(self, out_data, out_err):
        return out_err

class relu_layer(object):
    def forward(self, in_data):
        return np.maximum(in_data, np.zeros((1, in_data.shape[0]))).flatten()

    def backward(self, out_data, out_err):
        return (out_data > 0).astype(int) * out_err

class softmax_layer(object):
    def forward(self, in_data):
        temp_max = max(in_data)
        in_data -= temp_max   # avoid overflow
        exp_sum = sum(np.exp(in_data))
        return np.exp(in_data) / exp_sum

    def backward(self, out_data, out_err):
        jacobian = - np.outer(out_data, out_data)
        for _1 in range(out_data.shape[0]):
            jacobian[_1][_1] += out_data[_1]

        return np.dot(jacobian, out_err)
