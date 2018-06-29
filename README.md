To run nn_phylanx.physl in rostam, type below commands:

~/src/repos/phylanx/build_Release/bin/physl ~/experiment/nn_phylanx.physl 0.1 5000

, where
the 1st argument is the path of physl interpreter of phylanx release version,
the 2nd argument is the file name of physl version of neural network,
the 3rd argument is the learning rate (suggested: 0.1),
and the 4th argument is the iteration times (suggested: 5000).

The expected outcome should be closely to:
[[0.98570083]
 [0.97264431]
 [0.03657338]]


The original python code can be found:

https://www.analyticsvidhya.com/blog/2017/05/neural-network-from-scratch-in-python-and-r/



For the paramters, weights and biases(e.g. wh, bh, wout, bout), in nn_phylanx.physl, I randomly generated the values for these parameters and put the values inside the function, in order to reduce the amounts of argument inputs in the command line. I also used same values for weights and biases in nn_phython.py. Two filese should generate same results. 


