
from phylanx.ast import *
import numpy as np


@Phylanx(debug=True)
def main_phylanx(X, y, wh, bh, wout, bout, lr, num_iter):
    # create local variable
    wh_local = wh
    bh_local = bh
    wout_local = wout
    bout_local = bout
    output = 0
    hidden_layer_input1 = 0
    hidden_layer_input = 0
    hiddenlayer_activations = 0
    output_layer_input1 = 0
    output_layer_input = 0
    slope_hidden_layer = 0
    slope_output_layer = 0
    E = 0
    d_output = 0
    Error_at_hidden_layer = 0
    d_hiddenlayer = 0
    for i in range(num_iter):
        # forward
        hidden_layer_input1 = np.dot(X, wh_local)
        hidden_layer_input = hidden_layer_input1 + bh_local
        hiddenlayer_activations = (1 / (1 + np.exp(-hidden_layer_input)))
        output_layer_input1 = np.dot(hiddenlayer_activations, wout_local)
        output_layer_input = output_layer_input1 + bout_local
        output = (1 / (1 + np.exp(-output_layer_input)))

        # Backpropagation
        E = y - output
        slope_output_layer = (output * (1 - output))
        slope_hidden_layer = (hiddenlayer_activations * (1 - hiddenlayer_activations))
        d_output = E * slope_output_layer
        Error_at_hidden_layer = np.dot(d_output, np.transpose(wout_local))  # some problem
        d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
        wout_local += (np.dot(np.transpose(hiddenlayer_activations), d_output)) * lr
        bout_local += sum(d_output, 0, True) * lr
        wh_local += (np.dot(np.transpose(X), d_hiddenlayer)) * lr
        bh_local += sum(d_hiddenlayer, 0, True) * lr
    print(output, '\n')


# Variable initialization
# Input array
X = np.array([[1, 0, 1, 0], [1, 0, 1, 1], [0, 1, 0, 1]])

# Output
output_y = np.array([[1], [1], [0]])
num_iter = 5000  # Setting training iterations
lr = 0.1  # Setting learning rate
inputlayer_neurons = X.shape[1]  # number of features in data set
hiddenlayer_neurons = 3  # number of hidden layers neurons
output_neurons = 1  # number of neurons at output layer

# weight and bias initialization
wh = np.random.uniform(size=(inputlayer_neurons, hiddenlayer_neurons))
bh = np.random.uniform(size=(1, hiddenlayer_neurons))
wout = np.random.uniform(size=(hiddenlayer_neurons, output_neurons))
bout = np.random.uniform(size=(1, output_neurons))

main_phylanx(X, output_y, wh, bh, wout, bout, lr, num_iter)


