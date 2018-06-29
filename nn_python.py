import numpy as np
import time
def main(X, y, wh, bh, wout, bout, lr):
    wh_local = wh
    bh_local = bh
    wout_local = wout
    bout_local = bout
    output = 0
    for i in range(5000):

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
        bout_local += np.sum(d_output, axis=0, keepdims=True) * lr
        wh_local += (np.dot(np.transpose(X), d_hiddenlayer)) * lr
        bh_local += np.sum(d_hiddenlayer, axis=0, keepdims=True) * lr
    print(output)


# Input array
X = np.array([[1, 0, 1, 0], [1, 0, 1, 1], [0, 1, 0, 1]])

# Output
output_y = np.array([[1], [1], [0]])

# Variable initialization
epoch = 5000  # Setting training iterations
lr = 0.1  # Setting learning rate
inputlayer_neurons = X.shape[1]  # number of features in data set
hiddenlayer_neurons = 3  # number of hidden layers neurons
output_neurons = 1  # number of neurons at output layer

# weight and bias initialization
#wh = np.random.uniform(size=(inputlayer_neurons, hiddenlayer_neurons))
#bh = np.random.uniform(size=(1, hiddenlayer_neurons))
#wout = np.random.uniform(size=(hiddenlayer_neurons, output_neurons))
#bout = np.random.uniform(size=(1, output_neurons))
wh = np.array([[0.05414605, 0.02560007, 0.01744929],
               [0.82263347, 0.06131265, 0.201618],
               [0.77369734, 0.12512908, 0.58846846],
               [0.65657794, 0.18973852, 0.80577607]])

bh = np.array([[0.13638278, 0.9844139, 0.07151701]])

wout = np.array([[0.9507727],
                 [0.57245176],
                 [0.22638845]])
bout = np.array([[0.45447385]])
main(X, output_y, wh, bh, wout, bout, lr)


