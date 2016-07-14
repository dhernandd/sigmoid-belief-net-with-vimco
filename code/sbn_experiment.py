from sbn_vimco import *

# data loading capability
import fuel
fuel.config.data_path = '~/.kerosene/datasets/'
from keras.models import Sequential
from kerosene.datasets import binarized_mnist

#--------------
# keras makes our floatX = 'float32', which messes
# things up when we run on cpu for some reason
if theano.config.device == 'cpu':
    theano.config.floatX = 'float64'
#--------------

theano.config.optimizer = 'fast_run'

(X_train, ), (X_test, ) = binarized_mnist.load_data()

# flatten spatial dimensions of the data
X_train = X_train.reshape([-1,28*28])
X_test = X_test.reshape([-1,28*28])
# make validation set
X_valid = X_train[-10000:]
# reshape training set
X_train = X_train[:-10000]

X_train = X_train.astype(theano.config.floatX)
X_test = X_test.astype(theano.config.floatX)
X_valid = X_valid.astype(theano.config.floatX)

print(X_train.shape)
print(X_test.shape)
print(X_valid.shape)

xDim = 10    # number of top-level latents (one for each class)
yDim = 28*28 # number of output variables (one for each pixel)

# file name for output
fnam = 'sbn_best_validation_so_far.pkl'
# construct a dummy model 
gen_dict = dict({'layer_size':[100]})
rec_dict = dict({'layer_size':[100]})
model = VIMCO(gen_dict, SigmoidGenerative, rec_dict, SigmoidRecognition, xDim, yDim, filename = fnam, initial_patience = 10000)

costs = model.fit(X_train, y_valid=X_valid, batch_size = 24, max_epochs = 10, nSamp = 10, learning_rate=3e-3)
