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
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    num_classes_greater_margin = 0
    for j in range(num_classes):
      # Skip if images target class, no loss computed for that case.

      if j == y[i]:
        continue
        # Calculate  margin for svm, delta = 1
      margin = scores[j] - correct_class_score + 1 # note delta = 1

      # calculate loss and gradient if margin condition is violated.
      if margin > 0:
        num_classes_greater_margin += 1
        # Gradient for non correct class weight.
        dW[:, j] = dW[:, j] + X[i, :]
        loss += margin

    # Gradient for correct class weight.
    dW[:, y[i]] = dW[:, y[i]] - X[i, :] * num_classes_greater_margin

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  # Average our gradient across the batch and add gradient of regularization term.
  dW = dW / num_train + 2 * reg * W

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
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  num_train = X.shape[0]
  # s: A numpy array of shape (N, C) containing scores
  s = X.dot(W)
  # read correct scores into a column array of height N
  correct_score = s[list(range(num_train)), y]
  correct_score = correct_score.reshape(num_train, -1)
  # subtract correct scores from score matrix and add margin
  s += 1 - correct_score
  # make sure correct scores themselves don't contribute to loss function
  s[list(range(num_train)), y] = 0
  # construct loss function
  loss = np.sum(np.fmax(s, 0)) / num_train
  loss += reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  X_mask = np.zeros(s.shape)
  X_mask[s > 0] = 1
  X_mask[np.arange(num_train), y] = -np.sum(X_mask, axis=1)
  dW = X.T.dot(X_mask)
  dW /= num_train
  dW += 2 * reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
