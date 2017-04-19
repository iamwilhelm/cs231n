import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  scores = X.dot(W)
  num_examples = scores.shape[0]
  num_classes = scores.shape[1]
    
  # normalize scores with max score for numeric stabilitiy
  max_scores = np.max(scores, axis = 1)
  max_scores.shape = (num_examples, 1) # make into col vector
  max_scores = np.tile(max_scores, num_classes)
  norm_scores = scores - max_scores
  
  exp_scores = np.exp(norm_scores)
  probs = exp_scores / np.sum(exp_scores, axis = 1, keepdims = True)
  
  correct_logprobs = -np.log(probs[range(num_examples), y])

  data_loss = np.sum(correct_logprobs) / num_examples
  reg_loss = 0.5 * reg * np.sum(W * W)
  loss = data_loss + reg_loss
  
  
  dscores = probs
  dscores[range(num_examples), y] -= 1
  dscores /= num_examples
  
  dW = np.dot(X.T, dscores)
  dW += reg*W # don't forget the regularization gradient

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  num_examples = scores.shape[0]
  num_classes = scores.shape[1]

  # normalize scores with max score for numeric stabilitiy
  max_scores = np.max(scores, axis = 1)
  max_scores.shape = (num_examples, 1) # make into col vector
  max_scores = np.tile(max_scores, num_classes)
  norm_scores = scores - max_scores
  
  num_exp_score = np.exp(norm_scores)
  exp_scores = num_exp_score / np.sum(num_exp_score, axis = 1, keepdims = True)
  correct_logprobs = -np.log(exp_scores[range(num_examples), y])

  data_loss = np.sum(correct_logprobs) / num_examples
  reg_loss = 0.5 * reg * np.sum(W * W)
  loss = data_loss + reg_loss
  
  dscores = exp_scores
  dscores[range(num_examples), y] -= 1
  dscores /= num_examples
  
  dW = np.dot(X.T, dscores)
  dW += reg*W # don't forget the regularization gradient
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

