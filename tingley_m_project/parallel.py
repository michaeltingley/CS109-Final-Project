import math
from collections import defaultdict, OrderedDict
from random import uniform
import sys
import signal
import numpy as np
from copy import deepcopy
from pprint import pprint
import csv
from operator import itemgetter
from functools import partial
import base64
import pandas as pd

from scipy import sparse
import matplotlib.pyplot as plt
from sklearn import cross_validation, linear_model, gaussian_process, cluster, decomposition
import pickle

from mpi4py import MPI

ROOT_PROCESS_ID = 0

def read_csv_columns(filename):
  '''
  Reads in a csv of comma-delimited data. Returns a list of lists, where each
  list represents a column of the data in the csv.
  '''
  with open(filename, 'Ur') as f:
    return zip(*
      [x for x in csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)]
    )

def run_parallel_linear_regression(filename):
  '''
  Runs parallel multiple linear regression, inspired by a 1990 algorithm by Xu,
  Miller, and Wegman. This must be run on an MPI-enabled distributed cluster.

  Due to the distributed nature of this algorithm, instead of providing input
  data directly, this function requires input through a CSV file. This file is
  specified using the filename, the argument to this function. This file must
  have two or more columns, the last of which is the column for which prediction
  is desired.

  This function returns a tuple of (beta_0, beta_1). beta_0 is a matrix of a
  single number, the offset value of the regression. beta_1 is a column matrix
  of coefficients for the features of the input values.
  '''

  ''' PHASE 0: GET MPI CONFIGS '''

  mpi_communicator      = MPI.COMM_WORLD
  current_machine_index = mpi_communicator.Get_rank()
  number_of_machines    = mpi_communicator.Get_size()


  ''' PHASE 1: PRE-DISTRIBUTED PROCESSING ON ROOT PROCESSOR '''

  # The segments of the input and output data for use by each processor
  X_segments, y_segments = [[]] * number_of_machines, [[]] * number_of_machines
  # The total number of datapoints
  n = None

  # Serial processes by the root
  if current_machine_index == ROOT_PROCESS_ID:
    # This is the input data
    data = read_csv_columns(filename)
    Xs = zip(*data[:-1])
    ys = data[-1]
    # The number of datapoints
    n = len(Xs)
    # Number of data points on each machine (truncated)
    number_of_datapoints = n / number_of_machines
    # The remainder of datapoints that don't evenly fit onto the processors
    remainder_of_datapoints = n % number_of_machines

    # Partition the data into their cluster segments
    for machine_index in range(number_of_machines):
      # Compute the start and end indices of the data segments
      start = number_of_datapoints * machine_index + \
              min(machine_index, remainder_of_datapoints)
      end   = number_of_datapoints * machine_index + number_of_datapoints + \
              min(machine_index + 1, remainder_of_datapoints)

      # Grab the segment for the machine_index'th machine
      X_segments[machine_index] = np.matrix(Xs[start:end])
      y_segments[machine_index] = np.matrix(ys[start:end]).T


  ''' PHASE 2: DISTRIBUTE DATA AND PERFORM PARALLEL COMPUTATIONS '''

  # On each machine, grab n, the total number of entries
  n = mpi_communicator.bcast(n, root=ROOT_PROCESS_ID)
  # Scatter the segmented arrays to their respective processes
  # bold_X_k is the n_k-by-p matrix of features
  bold_X_k = mpi_communicator.scatter(X_segments, root=ROOT_PROCESS_ID)
  # bold_y_k is the n_k length vector of result values
  bold_y_k = mpi_communicator.scatter(y_segments, root=ROOT_PROCESS_ID)

  # n_k is the number of datapoints on this processor
  n_k = bold_X_k.shape[0]
  # bold_one is the n_k length vector of ones
  bold_one = np.matrix(np.ones((n_k, 1)))
  # bold_x_bar_k is the p length vector that is the average of the input columns
  bold_x_bar_k = bold_X_k.T * bold_one / n_k
  # y_bar_k is the number that is the average of the output column
  y_bar_k = bold_y_k.T * bold_one / n_k
  # bold_X_bar_k is the centered bold_X_k matrix
  bold_X_bar_k = bold_X_k - bold_one * bold_x_bar_k.T
  # bold_y_bar_k is the n_k length vector of output differences from the mean
  bold_y_bar_k = bold_y_k - bold_one * y_bar_k

  # The following are intermediate computations that need to be carried out
  bold_X_bar_trans_times_bold_X_bar_k = bold_X_bar_k.T * bold_X_bar_k
  bold_X_bar_trans_times_bold_y_bar_k = bold_X_bar_k.T * bold_y_bar_k
  # This value is not currently used -- we can use it to compute the sum of
  # squared error
  boly_y_bar_trans_times_bold_y_bar_k = bold_y_bar_k.T * bold_y_bar_k
  # Input responsibility is the weighted average of x values on this processor
  input_responsibility_k = n_k * bold_x_bar_k / n
  # Output responsibility is the weighted average of y values on this processor
  output_responsibility_k = n_k * y_bar_k / n

  # Note, below, that the reduce function adds together all values before
  # collecting them at the root. allreduce does the same, except collects them
  # on *all* processors.

  # Compute total input_responsibility for ALL processors. Conceptually, this is
  # bold_x_bar
  mpi_communicator.barrier()
  input_responsibility = mpi_communicator.allreduce(input_responsibility_k)

  # Compute total output_responsibility for ALL processors. Conceptually, this
  # is y_bar
  mpi_communicator.barrier()
  output_responsibility = mpi_communicator.allreduce(output_responsibility_k)

  # It's hard to explain what's going on here; please see the Findings document
  # for more details.
  bold_X_bar_trans_times_bold_y_bar_segment = \
    bold_X_bar_trans_times_bold_y_bar_k + \
    n_k * \
    (bold_x_bar_k - input_responsibility) * \
    (y_bar_k - output_responsibility)

  bold_x_bar_minus_input_responsibility = \
    bold_x_bar_k - input_responsibility

  bold_X_bar_trans_times_bold_X_bar_segment = \
    bold_X_bar_trans_times_bold_X_bar_k + \
    n_k * \
    bold_x_bar_minus_input_responsibility * \
    bold_x_bar_minus_input_responsibility.T


  ''' PHASE 3: COLLECT DATA & PERFORM SERIAL FOLLOW-UP COMPUTATIONS ON ROOT '''

  # Collect the results back at the root, reducing (which performs distributed
  # addition) wherever possible to enhance performance.
  mpi_communicator.barrier()
  bold_X_bar_trans_times_bold_y_bar = mpi_communicator.reduce(
    bold_X_bar_trans_times_bold_y_bar_segment,
    root=ROOT_PROCESS_ID
  )

  mpi_communicator.barrier()
  bold_X_bar_trans_times_bold_X_bar = mpi_communicator.reduce(
    bold_X_bar_trans_times_bold_X_bar_segment,
    root=ROOT_PROCESS_ID
  )

  if current_machine_index == ROOT_PROCESS_ID:
    beta_1 = bold_X_bar_trans_times_bold_X_bar.I * \
             bold_X_bar_trans_times_bold_y_bar
    beta_0 = output_responsibility - input_responsibility.T * beta_1
    return (beta_0, beta_1)

def run_parallel_neural_network(filename):
  '''
  Runs regression using a neural network. The specific algorithm is inspired by
  the Pattern Parallel Training algorithm, outlined by a 2008 paper by Dahl,
  McAvinney, and Newhall. This must be run on an MPI-enabled distributed
  cluster. This returns the resultant network on the root processor.
  '''

  ''' PHASE 0: CONFIGURATION '''

  mpi_communicator      = MPI.COMM_WORLD
  current_machine_index = mpi_communicator.Get_rank()
  number_of_machines    = mpi_communicator.Get_size()

  # This will ensure a little bit of overlap
  FRACTION_OF_SAMPLES_PER_MACHINE = min(1.5 / number_of_machines, 1.0)
  # Number of global epochs to run
  EPOCHS = 2500
  # Using a random seed allows us to assign the initial weights consistently
  # across all processors.
  RANDOM_SEED = 65316875
  # Parameters to the neural network
  HIDDEN_LAYERS = [4, 3]
  LEARNING_RATE = 0.2
  MOMENTUM_FACTOR = 0.15
  PRINT_PERIOD = 200


  ''' PHASE 1: PRE-DISTRIBUTED PROCESSING

  The algorithm requires that all data be present on all machines. Therefore,
  rather than have the root process read and distribute the data, we have the
  all processes read the data independently. Since these are reads, they can be
  done in parallel without any fear of data mangling or inconsistency.
  '''

  # Read the input data
  data = read_csv_columns(filename)
  # This is *all* of the features
  bold_X = np.matrix(zip(*data[:-1]))
  # This is *all* of the outputs
  bold_y = np.matrix(data[-1]).T
  # The size and dimension of all of the data
  N, D = bold_X.shape
  # The size of the data to process on this processor
  n_k = int(math.ceil(FRACTION_OF_SAMPLES_PER_MACHINE * N))

  # All random computations will now be globally-consistent
  np.random.seed(RANDOM_SEED)
  # Initialize the neural network randomly
  nn = NeuralNetwork(bold_X, bold_y, HIDDEN_LAYERS, train_immediately=False)
  # All random computations will now be globally-inconsistent
  np.random.seed()


  ''' PHASE 2: DISTRIBUTE DATA AND PERFORM PARALLEL COMPUTATIONS

  Here we want to run the distributed version of the neural network. This is
  done in a few steps, so we'll start slowly:

  We iterate for a number of epochs. This can be determined by the mean
  squared error, but in practice, since the error rate decays over time, we run
  for 'as long as possible', so it's reasonable to specify this for a number of
  epochs. Each iteration, we choose a random subset of the data to process.

  For each iteration, we want to record the weight updates for that iteration.
  This is so that we can broadcast these weight updates to the other processors.
  The serial neural network algorithm that we built doesn't quite do what we
  want, since it's optimized to run for many training trials. However, since
  we've exposed some of the otherwise internal neural network API, we can take
  over here to do what we want. Specifically, we want to record the weights
  before each training session, train, and then record the weights after. By
  subtracting the weights after training from the weights before training, we
  can get all of the weight update matrices for each layer. We can then
  broadcast and sum these matrices to all processors using MPI's AllReduce.
  These weight updates can be applied to each processes *PRE-TRAINING* weights
  to achieve a *globally-consistent* set of post-training weight matrices. It's
  also simple to record the new momentum matrices, since they literally are the
  update matrices that each processor is receiving.

  By repeating the above process, we can run multiple distributed training
  epochs of the NeuralNetwork. It's important that all of the distributed
  weight matrices remain globally consistent, so every few transmission
  iterations, we check to ensure that all of the weight matrices are, within a
  very small margin of error, the same as the root process's weight matrix.
  '''

  for i in range(EPOCHS):
    # Select a subset (a matrix) of length n_k from all of the data
    sample_indices = random.sample(range(N), n_k)
    # The inputs to use locally for this iteration
    bold_X_k = bold_X[sample_indices]
    # The corresponding outputs to use for this iteration
    bold_y_k = bold_y[sample_indices]

    # Grab the original weight matrices so we can compute the changes
    original_weight_matrices = deepcopy(nn.weight_matrices)
    # Now actually train for an iteration of the local data
    local_error = nn.run_training_iteration(
      bold_X_k,
      bold_y_k,
      LEARNING_RATE,
      MOMENTUM_FACTOR
    )
    # Compute the change in weight matrices
    change_in_weight_matrices = nn.weight_matrices - original_weight_matrices

    # Once all processors have finished the epoch, we globally sum the change in
    # weight matrices and apply them to the original weight matrices to retrieve
    # the globally-consistent weight matrices to use for the next iteration.
    mpi_communicator.barrier()
    total_weight_changes, total_error = mpi_communicator.allreduce(
      np.array([change_in_weight_matrices, local_error])
    )

    nn.weight_matrices = original_weight_matrices + \
                         total_weight_changes / number_of_machines

    if i % PRINT_PERIOD == 0:
      if current_machine_index == ROOT_PROCESS_ID:
        print 'Iteration', i, 'error:', total_error


  ''' PHASE 3: COLLECT DATA & PERFORM SERIAL FOLLOW-UP COMPUTATIONS

  At this stage, we have a globally-consistent set of weight matrices that have
  arisen from performing a distributed training of the NeuralNetwork. Since
  these weight matrices are globally consistent, we actually don't have any
  follow-up collection of distributed data to perform, and can use the computed
  weight matrices as a basis for the neural network.
  '''

  return nn if current_machine_index == ROOT_PROCESS_ID else None
