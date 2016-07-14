# Sigmoid Belief Networks (SBN) with VIMCO

This is a reference implementation of the algorithm described in:

* Mnih, Andriy, and Rezende, Danilo. "Variational inference for Monte Carlo objectives." arXiv:1602.06725. 2016.

and may be used to reproduce the MNIST results of that paper. The code is written in Python 2 using Theano with the Lasagne library. 

# Useage

Running `sbn_experiment.py` by default trains a 200-200-10 sigmoid belief network (a three-layer network with 200 units in the first layer, 200 units in the second layer, and 10 units in the 'top' layer). These parameters are easy to modify. 

The Keras library should automatically install and make available the binarized MNIST dataset. 

After each epoch (pass through the data), the code writes out a Pickle file that saves the current best state of the model; the file name is set by the `fnam` variable in `sbn_experiment.py`. 

The IPython notebook `test_sbn.ipynb` loads files (expecting the file name `sbn_best_validation_so_far.pkl`), and performs some simple visualizations of the fit model. 
