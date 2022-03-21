"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.

  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch

  TODO:
  Implement accuracy computation.
  """

  accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(targets, axis=1))/targets.shape[0]

  return accuracy

def train():
  """
  Performs training and evaluation of ConvNet model.

  TODO:
  Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)
  torch.manual_seed(42)

  # device configuration
  cuda = torch.cuda.is_available()
  if cuda:
      print('Running on GPU')
  device = torch.device('cuda' if cuda else 'cpu')
  dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor
  # load dataset
  cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)


  # initialize list for accuracies
  accuracies = []
  # initialize list for losses
  losses = []
  # initialize batches
  batches = []
  for i in range(FLAGS.max_steps):
      x, y = cifar10['train'].next_batch(FLAGS.batch_size)
      batches.append((x,y))
  # initialize the model
  output_size = batches[-1][1].shape[1] # output size
  input_size = batches[-1][0].shape[1] # input size  
  model = ConvNet(input_size, output_size).to(device)

  # intialize loss function
  criterion = nn.CrossEntropyLoss()
  # intialize optimizer
  optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate)
  
  # train the model
  for step in range(FLAGS.max_steps): # for each step
    x,t = batches[step] # training images and targets
    # forward pass
    y = model(torch.from_numpy(x).type(dtype)) # prediction
    t = torch.from_numpy(t).type(dtype)
    t = torch.max(t,1)[1] # actual label
    loss = criterion(y, t) # calculate loss
    losses.append(loss.cpu().detach().numpy())
    # backward pass
    optimizer.zero_grad()    
    loss.backward()
    optimizer.step()

    # test the model accuracy on test data, after every 500 steps
    if step % FLAGS.eval_freq == 0:
        # load test images and labels
        x, t = cifar10['test'].images, cifar10['test'].labels        
        with torch.no_grad(): # predict the labels of the test images
          y = model(torch.from_numpy(x).type(dtype))
        # calculate accuracy using the accuracy function
        acc = accuracy(y.cpu().detach().numpy(),t) 
        accuracies.append(acc) # add accuracy to the list
        print('ACCURACY at step',step,': ',acc)

  # Save a plot of the accuracies
  x_axis = list(range(0, MAX_STEPS_DEFAULT, EVAL_FREQ_DEFAULT))
  default_x_ticks = range(len(x_axis))  
  _ = plt.clf()
  _ = plt.plot(default_x_ticks, accuracies, label='accuracy', marker=".")
  _ = plt.xticks(default_x_ticks, x_axis)
  _ = plt.xlabel("Steps")
  _ = plt.ylabel("Accuracy")
  _ = plt.legend() 
  _ = plt.savefig('_accuracies')

  # Save a plot of the losses
  _ = plt.clf()
  _ = plt.plot(losses, label='loss')
  _ = plt.xlabel("Steps")
  _ = plt.ylabel("Loss")
  _ = plt.legend() 
  _ = plt.savefig('_loss_curve')


def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()

  main()