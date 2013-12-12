import math
from random import uniform
import numpy as np
from copy import deepcopy
import sys
import signal
import pickle

from collections import defaultdict

class NeuralNetwork(object):
  '''
  Serial neural networks using back-propagation with momentum. Uses an arbitrary
  number of completely-connected hidden layers of arbitrary numbers of nodes
  each. Uses an additional input node to represent bias.

  Design Notes
  ------------
  After initialization, all layers are represented in the same way.
  For instance, all activation values are represented in one list of lists,
    which includes the input, hidden, and output nodes.
  All weights are stored in a single list of numpy matrices, where each matrix
    represents the weights from one hidden layer (or the input layer) to the
    next hidden layer (or output layer).
  We construct the NeuralNetwork so that it has an additional input node at the
    'bottom' (last input node), which represents the bias term.
  '''

  ''' The sigmoid function for activation threshold. '''
  vectorized_sigmoid = np.vectorize(math.tanh)
  '''
  The derivative of the sigmoid function, for back propagation.
  Defined as sigmoid(x) * (1 - sigmoid(x)).

  NOTE: The input value to this should be the value of sigmoid(x), and not x
  '''
  vectorized_dsigmoid = np.vectorize(lambda sig_of_x: 1.0 - sig_of_x ** 2)

  ''' True if the element could represent a normalized value. '''
  vectorized_is_normalized = np.vectorize(lambda x: x <= 1.0 and x >= 0.0)

  ''' The square function. '''
  vectorized_square = np.vectorize(lambda x: x ** 2)

  def __init__(
      self,
      training_patterns,
      outputs,
      hidden_layers,
      weight_initialization_function=lambda layer, num_nodes, num_nodes_next:
        np.matrix(np.random.uniform(-0.2, 0.2,(num_nodes, num_nodes_next))),
      iterations=1000,
      learning_rate=0.2,
      momentum_factor=0.15,
      print_period=100,
      train_immediately=True,
      on_interrupt_save_filename=None):
    '''
    Initializes and trains the neural network with the specified training
    patterns and outputs, using the network structure implied by the hidden
    layers.

    Parameters
    ----------
    training_patterns: numpy matrix. Each row represents an input pattern.
      IMPORTANT NOTE: The input patterns must be normalized; that is, they must
      represent values between 0 and 1 in order for training to work correctly.
    outputs: numpy matrix. The ith row represents the outputs for the ith
      training pattern. IMPORTANT NOTE: The outputs must be normalized; that is,
      they must represent values between 0 and 1 in order for training to work
      correctly.
    hidden_layers: List of integers. The length of the list is the number of
      hidden layers. The ith integer in the list is the number of hidden nodes
      in the ith hidden network layer.
    weight_initialization_function: A function from the current layer number,
      the number of nodes in this layer, and the number of nodes in the next
      layer to a numpy matrix, whose dimensions are the number of nodes in this
      layer by the number of nodes in the next layer. This function determines
      the weight initializations for the edges from each layer to each next
      layer.
    iterations: The integer number of iterations to go through all of the data.
      One iteration consists of feeding forward all of the data and back
      propagating the error rates.
    learning_rate: The coefficient in [0, 1] that is the rate to learn from
      back propagation errors.
    momentum_factor: The coefficient in [0, 1] that we want to retain the old
      weights when computing the new weights in back propagation.
    print_period: The int value that determines how many iterations to wait
      before printing updates to stdout, or None to disable.
    train_immediately: True if the NeuralNetwork should train itself using the
      provided data as a part of initialization.
    on_interrupt_save_filename: The pickle file to write this NN to when ctrl+C
      is pressed. The network will finish its current training iteration before
      saving.
    '''
    assert len(training_patterns) > 0, \
      'At least one training pattern must be provided'
    assert len(training_patterns) == len(outputs), \
      'There must be the same number of training patterns and output values'
    assert len(hidden_layers) > 0, \
      'hidden_layers must be a non-empty list of ints'

    assert np.all(NeuralNetwork.vectorized_is_normalized(training_patterns)), \
      'Training patterns must be normalized to be between 0.0 and 1.0'
    assert np.all(NeuralNetwork.vectorized_is_normalized(outputs)), \
      'Outputs must be normalized to be between 0.0 and 1.0'

    assert isinstance(iterations, int) and iterations > 0, \
      'Iterations must be a positive int'
    assert learning_rate < 1 and learning_rate > 0, \
      'The learning rate must be between 0 and 1 (exclusive)'
    assert momentum_factor < 1 and momentum_factor > 0, \
      'The learning rate must be between 0 and 1 (exclusive)'

    # Number of training examples
    N = training_patterns.shape[0]
    # Number of data dimensions
    D = training_patterns.shape[1] + 1
    # Number of output dimensions
    OUTPUT_DIMENSIONS = outputs.shape[1]

    # Initialize activations
    # List of activations for each input, hidden, and output node
    self.activations = [np.matrix([None] * num) \
                        for num in [D] + hidden_layers + [OUTPUT_DIMENSIONS]]

    assert len(self.activations) is 1 + len(hidden_layers) + 1, \
      'The number of layers was constructed incorrectly'
    # Check that the number of nodes is correct
    assert sum(a.size for a in self.activations) == \
      D + sum(hidden_layers) + OUTPUT_DIMENSIONS, \
      'The number of nodes was constructed incorrectly'

    # A list of matrices of the weights in the neural network. For instance, the
    # first element of this list will be a (D x hidden_layers[0]) matrix of the
    # input-to-first-hidden-layer weights. The last element of this list will be
    # a (hidden_layers[-1] x OUTPUT_DIMENSIONS) matrix of the last-hidden-layer-
    # to-output weights.
    self.weight_matrices = \
      np.array(
        list(
          weight_initialization_function(
            i,
            self.activations[i].size,
            self.activations[i + 1].size
          ) for i in range(len(self.activations) - 1)
        )
      )
    assert len(self.weight_matrices) == len(self.activations) - 1, \
      'The number of weight matrices was constructed incorrectly'
    for i in range(len(self.weight_matrices)):
      assert self.weight_matrices[i].shape[0] == self.activations[i].size, \
        'The # rows of weight matrix ' + str(i) + ' was constructed incorrectly'
      assert self.weight_matrices[i].shape[1] == self.activations[i + 1].size, \
        'The # cols of weight matrix ' + str(i) + ' was constructed incorrectly'

    self.momentum_matrices = np.array(
      [np.zeros(weight_matrix.shape) for weight_matrix in self.weight_matrices]
    )
    assert len(self.momentum_matrices) == len(self.weight_matrices), \
      'The number of momentum_matrices matrices was constructed incorrectly'
    for i in range(len(self.momentum_matrices)):
      assert self.weight_matrices[i].shape == self.weight_matrices[i].shape, \
        'Momentum matrix ' + str(i) + ' had the incorrect shape'

    self.on_interrupt_save_filename = on_interrupt_save_filename

    if train_immediately:
      self.train(
        training_patterns,
        outputs,
        iterations,
        learning_rate,
        momentum_factor,
        print_period
      )

  def train(
      self,
      training_patterns,
      targets,
      iterations,
      learning_rate,
      momentum_factor,
      print_period):
    '''
    Trains the network by repeatedly feeding forward the inputs and back
    propagating the errors.
    '''
    interrupted = [False]
    def set_interrupted(_, __):
      interrupted[0] = True
    signal.signal(signal.SIGINT, set_interrupted)

    self.training_history = []

    for i in xrange(iterations):
      if interrupted[0]:
        print 'Interrupt detected, saving neural network as', \
          self.on_interrupt_save_filename, 'and quitting'
        pickle.dump(self, open(self.on_interrupt_save_filename, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        sys.exit()

      error = self.run_training_iteration(
        training_patterns,
        targets,
        learning_rate,
        momentum_factor
      )
      self.training_history.append(error / len(training_patterns))

      if print_period is not None and i % print_period == 0:
        print 'Iteration', i, 'MSE:', self.training_history[i]

  def run_training_iteration(
      self,
      training_patterns,
      targets,
      learning_rate,
      momentum_factor):
    '''
    Runs an iteration of training for the provided training_patterns and
    targets using the provided learning_rate and momentum_factor. This does
    result in the weights and momentum matrices being updated.

    This function returns the total error after running the training iteration.
    '''
    total_error = 0.0

    for pattern, target in zip(training_patterns, targets):
      # Feed forward in order to compute the NeuralNetwork delta matrices
      self.feed_forward(pattern)
      # Run back propagation, which will update the weights
      self.back_propagate(target, learning_rate, momentum_factor)
      total_error += self.compute_error_for_targets(target)

    return total_error

  def feed_forward(self, inputs):
    '''
    Takes a matrix of inputs and feeds them forward through the network.

    Returns the new activation values for the output nodes.
    '''
    # Convert it to a matrix
    # + [1.0] because of the bias node
    inputs = np.append(inputs, np.matrix(1.0), 1)

    assert inputs.shape == self.activations[0].shape, \
      'The input shape and the input node layer shape were different'

    # Set the input activations
    self.activations[0] = inputs

    # Feed forward through the hidden layers
    # For each layer...
    for layer in range(1, len(self.activations)):
      # Update its activation value by summing over its weighted inputs
      self.activations[layer] = NeuralNetwork.vectorized_sigmoid(
        self.activations[layer - 1] * self.weight_matrices[layer - 1]
      )

    return self.activations[-1]

  def back_propagate(self, targets, learning_rate, momentum_factor):
    '''
    Runs back propagation for the currently computed activations and weights for
    the target output values.
    '''
    assert targets.shape[0] == 1, 'back_propagate handles one output at a time'
    assert targets.shape[1] == self.activations[-1].size, \
      'The number of target values did not match the number of output nodes'

    # Get the deltas
    deltas = self.compute_deltas(targets)
    self.update_weights(deltas, learning_rate, momentum_factor)

  def compute_deltas(self, targets):
    '''
    Computes the deltas used in the neural network from the target values and
    the activation values currently on the network.

    For the output layer, the deltas are the difference between the output node
    activation values and the actual target values.

    For each layer in turn before the output layer, the deltas for node i are
    the sum of the deltas in the next layer, weighted by the edge going from i
    to each next node.

    This function returns a list of row matrices. The length of the list is the
    number of layers in the network. The ith matrix element of the jth list
    element is the computed delta for the ith node in the jth layer.
    '''
    # The ith delta is a row matrix representing the deltas for the node weights
    # in the ith layer of the NeuralNetwork
    deltas = [None] * len(self.activations)

    # Calculate error and delta terms for the last layer
    # errors is the difference between the target values and the activations
    errors = targets - self.activations[-1]
    deltas[-1] = np.multiply(
      NeuralNetwork.vectorized_dsigmoid(self.activations[-1]),
      errors
    )

    # For each layer in reverse, propagate back and compute the deltas. We don't
    # need to compute the deltas for the input layer.
    for layer in reversed(range(1, len(self.activations) - 1)):
      # Errors is the sum of {the weights outgoing from each node times the
      # delta of the node that they're going to}
      errors = deltas[layer + 1] * self.weight_matrices[layer].T
      # np.multiply is element-wise multiplication
      deltas[layer] = np.multiply(
        NeuralNetwork.vectorized_dsigmoid(self.activations[layer]),
        errors
      )

    return deltas

  def update_weights(self, deltas, learning_rate, momentum_factor):
    '''
    Given the deltas, this updates the weight matrices to reflect the deltas.

    Each weight update is computed like:
      weight from i in layer to j in layer+1 is:
        change <- deltas[layer][]
    '''
    # Input validation
    assert len(deltas) == len(self.activations), \
      'Different number of deltas and layers'
    for layer in range(1, len(self.activations)):
      assert deltas[layer].shape == self.activations[layer].shape, \
        'Different shape for deltas and network in layer ' + str(layer)

    # For each layer, update the weights according to the deltas, learning rate,
    # and momentum
    for layer, weight_matrix in enumerate(self.weight_matrices):
      # Compute the change matrix, which is each activation multiplied by each
      # delta. E.g., the top row will be the first node's activation multiplied
      # in turn by each delta (realize delta_i corresponds to weight_i).
      change_matrix = self.activations[layer].T * deltas[layer + 1]
      self.weight_matrices[layer] += \
        learning_rate * change_matrix + \
        momentum_factor * self.momentum_matrices[layer]
      self.momentum_matrices[layer] = change_matrix

  def compute_error_for_targets(self, targets):
    '''
    Computes and returns the mean squared prediction error for the targets.
    '''
    return np.sum(self.vectorized_square(targets - self.activations[-1]))

  def get_training_history(self):
    '''
    Returns the training history of this neural network. The training history
    is a list where the ith element is the MSE of the ith epoch.
    '''
    return self.training_history

  def predict(self, input):
    '''
    Predicts and returns the output values for the given input values. input
    should be either a one dimensional array or a row matrix.
    '''
    return self.feed_forward(np.matrix(input))
