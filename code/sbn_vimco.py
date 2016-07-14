import theano
import theano.tensor as T
import theano.tensor.nlinalg as Tla
import lasagne       # the library we're using for NN's
# import the nonlinearities we might use 
from lasagne.nonlinearities import leaky_rectify, softmax, linear, tanh, rectify, sigmoid
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np
from numpy.random import *
from matplotlib import pyplot as plt

import cPickle
import sys
import dill
import collections

import scipy

# Load our code

# Add all the paths that should matter right now
sys.path.append('lib/') 
from MinibatchIterator import *
from GenerativeModel import *       # Class file for generative models. 
from RecognitionModel import *      # Class file for recognition models
#from VIMCO import *

from theano.compile.nanguardmode import NanGuardMode


class Trainable(object):
    ''' This class does useful things during training: record stats, save the model, etc.'''

    def __init__(self, do_validation_save = True, filename = None, initial_patience = 10000):

        # early-stopping parameters
        # compute the 'patience', borrowed from the deep learning tutorial
        self._patience = initial_patience  # look as this many examples regardless
        self._patience_increase = 2  # wait this much longer when a new best is found
        self._improvement_threshold = 0.995  # a relative improvement of this much is
                                             # considered significant

        # filename for saving out
        self._filename = filename
    
        # iteration counter
        self._iter_count = 0
        
        ## variables for measuring time spent in optimization
        self._epochs_since_improvement = 0
        self._runtime_iter = collections.OrderedDict([])

        # lists that record our optimization history
        self._costs = collections.OrderedDict([])
        self._validation_costs = collections.OrderedDict([])

        # the elbo value of our "best" model
        self._best_validation_cost = -np.inf
        
        self._SAVE_ON_VALIDATION_IMPROVEMENT = do_validation_save
        
    def out_of_patience(self):        
        ''' Return True if we've exceeded our patience, False otherwise.'''
        if self._patience <= self._iter_count:
            return True
        else:
            return False
        
    def append_log(self, cost, validation_cost = None, iteration_time = None):
        ''' 
            Call after a successful SGD iteration to append to log. TODO: add ability to save parameters & such.
        
            There's nothing forcing you to use this in a coherent way, so take care. 
        '''
        # increment iteration counter (we never have a 0th iteration)
        self._iter_count += 1 
        
        self._costs[self._iter_count] = cost
        # Save validation set error (if included)
        if validation_cost:
            self._validation_costs[self._iter_count] = validation_cost    
            # update our best-recorded valdiation error, and save parameters (if enabled)
            if validation_cost > self._best_validation_cost:
                # update the patience
                if validation_cost > self._best_validation_cost * self._improvement_threshold:
                    self._patience = max(self._patience, self._iter_count * self._patience_increase)
                # now update our record of the best-yet validation error
                self._best_validation_cost = validation_cost
                if self._SAVE_ON_VALIDATION_IMPROVEMENT and self._filename:
                    self.save_object(self._filename)
                    
        if iteration_time:
            self._runtime_iter[self._iter_count] = iteration_time
        
    def getParams(self):
        raise Exception('This is an intantiation of an abstract Trainable.')

    def save_object(self, filename):
        ''' Save object to file filename '''
        print('\t| Writing model to: %s.\n' % filename)        
        f = file(filename, 'wb')
        cPickle.dump(self, f)
        f.close()

    @classmethod
    def load_object(cls, filename):
        f = file(filename, 'rb')
        loaded_obj = cPickle.load(f)
        f.close()
        return loaded_obj

    def __getstate__(self):
        ''' Return object state. Use this as a template for modifying the save/load behavior of pickle.'''
        state = dict(self.__dict__)
        #1del state['training_set']
        return state

    def __setstate__(self, d):
        ''' Return object state. Use this as a template for modifying the save/load behavior of pickle.'''
        self.__dict__.update(d)
        #self.training_set = cPickle.load(file(self.training_set_file, 'rb'))


class SigmoidBernoulli(lasagne.layers.DenseLayer):
    ''' A Bernoulli "spiking" neural network layer.'''

#    def __init__(self, incoming, num_units, W=lasagne.init.Normal(), b=lasagne.init.Constant(-3.0), **kwargs):
    #def __init__(self, incoming, num_units, W=lasagne.init.Orthogonal(), b=lasagne.init.Constant(), **kwargs):
    def __init__(self, incoming, num_units, W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(-3.0), **kwargs):
        super(SigmoidBernoulli, self).__init__(incoming, num_units, W, b, nonlinearity = lasagne.nonlinearities.sigmoid, **kwargs)
        self._srng = RandomStreams()#np.random.randint(1, 2147462579))

    def get_output_for(self, input, deterministic=False, **kwargs):
        """
        Parameters
        ----------
        input : tensor
            output from the previous layer
        deterministic : bool
            If true noise is disabled, see notes
        """
        probs = T.clip(super(SigmoidBernoulli, self).get_output_for(input, **kwargs), 1e-7, 1-1e-7)
        
        if deterministic:
            return probs
        else:
            return self._srng.binomial(p=probs)   
        
class SigmoidGenerative(GenerativeModel):
    '''
    xDim - # classes
    yDim - dimensionality of observations
    '''
    def __init__(self, GenerativeParams, xDim, yDim):

        super(SigmoidGenerative, self).__init__(GenerativeParams,xDim,yDim)

        layer_size = GenerativeParams['layer_size']
        
        self.un_base_bias = theano.shared(value=np.ones([1,xDim]).astype(theano.config.floatX))
        self.base_bias = sigmoid(self.un_base_bias)
        
        sbn_nn = lasagne.layers.InputLayer((None, xDim))
        for ls in layer_size:
            sbn_nn = SigmoidBernoulli(sbn_nn, ls)
        self.sbn_nn = SigmoidBernoulli(sbn_nn, yDim)
        
    def sampleXY(self,_N, X = None):    
        ''' Optionally pass in X to set the top layer by hand.'''
        if X is None:
            # sample base 
            np_base_bias = np.asarray(self.base_bias.eval(), dtype=theano.config.floatX)        
            X = (np.random.rand(_N, self.xDim) > np.repeat(np_base_bias, _N, axis=0)).astype(theano.config.floatX)
        # sample down the network
        Y = lasagne.layers.get_output(self.sbn_nn, inputs = X)        
        return [X.astype('int32'), Y.eval()]
    
    def getSample(self, X, nSamp): 
        ''' Takes X, the top-layer input. '''        
        blah = []
        for idx in np.arange(nSamp):
             blah.append(lasagne.layers.get_output(lasagne.layers.get_all_layers(self.sbn_nn), inputs = X[idx]))
        return [T.stack(z) for z in zip(*blah)]
        
    def getParams(self):
        return [self.un_base_bias] + lasagne.layers.get_all_params(self.sbn_nn)

    def evaluateLogDensity(self, *hsamp):
        all_layers = lasagne.layers.get_all_layers(self.sbn_nn)
        # Get predictions given sampled inputs for h_{t} | h_{t-1} for t = n, n-2, 1
        pt = [lasagne.layers.get_output(lp, inputs = {lm:ym}, deterministic=True) for (lm, lp, ym) in zip(all_layers[:-1], all_layers[1:], hsamp[:-1])]
    
        def base_prob(h):
            return lasagne.objectives.binary_crossentropy(self.base_bias, h).sum(axis=-1)
        bb,_ = theano.map(base_prob, sequences=[hsamp[0]])
                
        bce = T.sum([lasagne.objectives.binary_crossentropy(p, y).sum(axis=-1) for (p, y) in zip(pt, hsamp[1:])],axis=0)
        ''' We sum here across layers, so that the output is number of samples by number of minibatches. '''
        return - bce - bb[:,0]
    
class SigmoidRecognition(RecognitionModel):

    def __init__(self,RecognitionParams,Input,xDim,yDim):
        '''
        An 'inverse' sigmoid belief network, mapping 'Input' observations into latent binary features. 
        
        layer_size - list of layer sizes, where [yDim -> layer_size[0] -> ... -> layer_size[-1] -> xDim]
        '''
        super(SigmoidRecognition, self).__init__(Input,None,xDim,yDim)
        
        layer_size = RecognitionParams['layer_size']
        
        sbn_nn = lasagne.layers.InputLayer((None, yDim))
        for ls in layer_size:
            sbn_nn = SigmoidBernoulli(sbn_nn, ls)
        self.sbn_nn = SigmoidBernoulli(sbn_nn, xDim)
        
    def getParams(self):
        network_params = lasagne.layers.get_all_params(self.sbn_nn)
        return network_params

    def getSample(self, Y, nSamp = 1):    
        def get_layers(ii):
            output = lasagne.layers.get_output(lasagne.layers.get_all_layers(self.sbn_nn), inputs = Y)
            return output[::-1]
        
        output,_ = theano.map(get_layers, T.arange(nSamp))
        return output

    def evalLogDensity(self, *hsamp_rev):
        hsamp = hsamp_rev[::-1]
        all_layers = lasagne.layers.get_all_layers(self.sbn_nn)
        # Get predictions given sampled inputs for h_{t} | h_{t-1} for t = n, n-2, 1
        pt = [lasagne.layers.get_output(lp, inputs = {lm:ym}, deterministic=True) for (lm, lp, ym) in zip(all_layers[:-1], all_layers[1:], hsamp[:-1])]

        ''' We sum here across layers, so that the output is number of samples by number of minibatches. '''
        return -T.sum([lasagne.objectives.binary_crossentropy(p, y).sum(axis=-1) for (p, y) in zip(pt, hsamp[1:])], axis=0)

class VIMCO(Trainable):
    def __init__(self, 
                gen_params, # dictionary of generative model parameters
                GEN_MODEL,  # class that inherits from GenerativeModel
                rec_params, # dictionary of approximate posterior ("recognition model") parameters
                REC_MODEL, # class that inherits from RecognitionModel
                xDim=2, # dimensionality of latent state
                yDim=2, # dimensionality of observations
                **kwargs
                ):
        
        super(VIMCO, self).__init__(**kwargs)
        
        #---------------------------------------------------------
        ## actual model parameters
        self.Y = T.matrix('Y')   # symbolic variables for the data

        self.xDim   = xDim
        self.yDim   = yDim
        
        # instantiate our prior & recognition models
        self.mrec   = REC_MODEL(rec_params, self.Y, self.xDim, self.yDim)
        self.mprior = GEN_MODEL(gen_params, self.xDim, self.yDim)

        self.isTrainingRecognitionModel = True;
        self.isTrainingGenerativeModel = True;
        
    def getParams(self):
        ''' 
        Return Generative and Recognition Model parameters that are currently being trained.
        '''
        params = []        
        params = params + self.mprior.getParams()            
        params = params + self.mrec.getParams()            
        return params        
        
    def compute_objective_and_gradients(self, nSamp):
        hsamp = self.mrec.getSample(self.Y, nSamp)

        # evaluate the generative model density P_\theta(y_i , h_i)
        p_yh,_ = theano.map(self.mprior.evaluateLogDensity, sequences=hsamp)
        # evaluate the recognition model density Q_\phi(h_i | y_i)
        q_hgy,_ = theano.map(self.mrec.evalLogDensity, sequences=hsamp)

        ff = (p_yh-q_hgy)
        sortidx = ff.argsort(axis=0)
        
        fmax = ff[(sortidx[-1],T.arange(ff.shape[-1]))].dimshuffle('x',0)
        
        f_hy = T.exp(ff - fmax)
        sum_across_samples = f_hy.sum(axis=0, keepdims = True)
        Lhat = T.log(sum_across_samples/nSamp) + fmax
         
        col_idx = T.arange(ff.shape[-1])
                                                                          # This 1e-12 constant is for debugging nans
                                                                          # in other parts of code. We know we'll get 
                                                                          # nans where we'll then overwrite. Use it with 
        # compute cross-validated estimates of Lhat                       # nanguard mode.
        hold_out_except_last = T.log((sum_across_samples - f_hy)/(nSamp-1)) + fmax #+1e-12) + fmax
        f2max_vec = ff[(sortidx[-2],T.arange(ff.shape[-1]))]
        f2max = f2max_vec.dimshuffle('x',0)
        # Do tricky things to keep the numerics in order (avoid a term being \approxeq 0)
        ff_nolast = T.set_subtensor(ff[(sortidx[-1],col_idx)], f2max_vec)
        f_hy_last = T.exp(ff_nolast - f2max)
        # compute held-out sum when we hold out the maximum element
        hold_out_last = T.log((f_hy_last.sum(axis=0, keepdims=True) - f_hy_last)/(nSamp-1)) + f2max    
        # compute final held-out estimates
        hold_out = T.set_subtensor(hold_out_except_last[(sortidx[-1],col_idx)], hold_out_last[(sortidx[-1],col_idx)])
                        
        Lhat_cv = Lhat - hold_out 
        the_ws = f_hy / sum_across_samples

        weighted_q = T.sum((Lhat_cv*q_hgy + the_ws*ff).mean(axis=1))
        #weighted_q = T.sum((Lhat_cv*q_hgy + the_ws*(p_yh-q_hgy)).sum(axis=1))

        # gradients for approximate posterior
        dqhgy = T.grad(cost=weighted_q, wrt = self.mrec.getParams(), consider_constant=([the_ws,Lhat_cv]+hsamp), disconnected_inputs='ignore')
        
        # gradients for prior
        dpyh = T.grad(cost=T.sum((the_ws*ff).mean(axis=1)), wrt = self.mprior.getParams(), consider_constant=hsamp + [the_ws], disconnected_inputs='ignore')        
        #dpyh = T.grad(cost=T.sum((the_ws*(p_yh-q_hgy)).sum(axis=1)), wrt = self.mprior.getParams(), consider_constant=hsamp + [the_ws], disconnected_inputs='ignore')        
        
        return [Lhat.mean(), dpyh, dqhgy]
    
    def update_params(self, grads, L):
        batch_y = T.matrix('batch_y')
        lr = T.scalar('lr')

        # SGD updates
        #updates = [(p, p + lr*g) for (p,g) in zip(self.getParams(), grads)]

        # Adam updates        
        # We negate gradients because we formulate in terms of maximization.
        updates = lasagne.updates.adam([-g for g in grads], self.getParams(), lr) 
        
        perform_updates_params = theano.function(
                 outputs=L,
                 inputs=[ theano.In(batch_y), theano.In(lr)],
                 updates=updates,
                 givens={
                     self.Y: batch_y,
                 }#,
#                 mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True)
             )

        return perform_updates_params

    def fit(self, y_train, y_valid = None, batch_size = 50, max_epochs=100, learning_rate = 3e-4, nSamp = 5, validation_frequency = None):
        import time  # for timing iterations

        if not validation_frequency:
            validation_frequency = np.floor(y_train.shape[0]/np.asarray(batch_size, dtype='int32'))

        if y_valid is not None:
            print('\n\tEvaluating on validation set every %d iterations\n.' % validation_frequency)
        
        train_set_iterator = IIDDatasetIterator(y_train, batch_size)        
                    
        [Lhat, dpyh, dqhgy] = self.compute_objective_and_gradients(nSamp)
        
        param_updater = self.update_params(dpyh+dqhgy,Lhat)
            
        valid_cost_fn = self.get_valid_set_evaluator(Lhat, y_valid, batch_size, duty = validation_frequency)
  
        epoch = 0
        while epoch < max_epochs and (not self.out_of_patience()):
            sys.stdout.write("\r%0.2f%%\n" % (epoch * 100./ max_epochs))
            sys.stdout.flush()
            batch_counter = 0
            for y in train_set_iterator:
                t0 = time.time()
                avg_cost = param_updater(y, learning_rate)
                t1 = time.time()
                if np.mod(batch_counter, 10) == 0:
                    print('(ii,L): (%d,%f)\n' % (batch_counter, avg_cost))
                self.append_log(cost=avg_cost, validation_cost = valid_cost_fn(self._iter_count), iteration_time = t1 - t0)
                batch_counter += 1
            epoch += 1
        return self._costs
    
    def get_valid_set_evaluator(self, cost, dataset, batch_size, duty = 100):
        ''' duty - duty cycle of validation evaluations. returns training function that evaluates to None except when 
                    the input index is 0 mod duty
        '''
        # if we have no validation set, return a function that always evaluates to 'None'
        if dataset is None: 
            def nonefn():
                return None
            return nonefn
        # ---------------------------
        # otherwise, we want a validation set cost evaluator
        valid_set_iterator = IIDDatasetIterator(dataset, batch_size)
        batch_y = T.matrix('batch_y')
        evaluate_cost = theano.function(inputs=[theano.In(batch_y)],
                                outputs=cost,
                                givens={self.Y: batch_y})
        def average_validation_cost(ii):
            if (ii+1) % duty:
                return None
            else:
                return np.mean([evaluate_cost(batch_y) for batch_y in valid_set_iterator])
        return average_validation_cost
