import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    num_outside_margin = 0
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        num_outside_margin += 1
        dW[:, j] += 1 * X[i]
        
    dW[:, y[i]] += -num_outside_margin * X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += (reg * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]

  scores = X.dot(W)
  #print "scores: ", scores[0:2]

  scores_y = scores[np.arange(num_train), np.array(y)]
  scores_y.shape = (num_train, 1)
  tiled_scores_y = np.tile(scores_y, (1, num_classes))

  margin = scores - tiled_scores_y + 1
  margin[np.arange(num_train), np.array(y)] = 0  # the correct class has no penalty
  #print "margin: ", margin.shape
  
  dataloss = np.maximum(margin, np.zeros([num_train, num_classes]))
  #print "dataloss: ", dataloss
  
  normalizedloss = np.sum(dataloss) / num_train
  #print "normalized loss: ", normalizedloss
  
  regloss = 0.5 * reg * np.sum(W * W)
  #print "reg loss: ", regloss

  loss = normalizedloss + regloss
  #print "loss: ", loss



  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  
  # margin = scores - tiled_scores_y + 1
  num_outside_margin = (margin > 0).sum(axis = 1) - 1 # subtract 1 to exclude correct class
  dM = np.zeros((num_train, num_classes)) # classes with margin < 0 has dW of 0
  dM[margin > 0] = 1 # classes with margin > 0 has dW of 1

  # put the -#m weights in the correct class for each training example
  dM[range(num_train), y] = -num_outside_margin
  dM /= num_train # normalize by number of training examples

  # DL = Data Loss
  dDL = X.T.dot(dM)
  
  # RL = Regularization Loss
  dRL = reg * W
  
  dW = dDL + dRL
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
