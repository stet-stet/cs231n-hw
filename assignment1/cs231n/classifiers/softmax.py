from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_batch = X.shape[0]
    dim = X.shape[1]
    classes = W.shape[1]
    for i in range(num_batch):
      data = X[i]
      s = data.dot(W)
      dLdS = np.zeros(classes)
      sum_of_exps = 0
      for j in range(classes):
        sum_of_exps += np.exp(s[j])
      loss += (-np.log(np.exp(s[y[i]])/sum_of_exps))
      for j in range(classes):
        if j==y[i]:
          dLdS[j] = -1 + np.exp(s[y[i]])/sum_of_exps
        else:
          dLdS[j] = np.exp(s[j])/sum_of_exps
      dW += np.outer(data,dLdS)
    loss /= num_batch
    loss += reg*np.sum(W*W)
    dW /= num_batch
    dW += 2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    dim = X.shape[1]
    classes = W.shape[1]
    s = np.matmul(X,W)
    s -= np.max(s,axis=1)[:,np.newaxis]
    ex = np.exp(s)
    sum_ex = np.sum(ex,axis=1)
    loss = -np.mean(np.log ( ex[np.arange(num_train),y] / sum_ex))

    dLdS = ex / sum_ex[:,np.newaxis] 
    dLdS[np.arange(num_train),y] -= 1

    dW = np.mean(np.matmul(X[:,:,np.newaxis],dLdS[:,np.newaxis,:]),axis=0)
    dW += 2 * reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
