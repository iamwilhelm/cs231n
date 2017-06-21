import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
  """
  Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
  activation function.

  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.

  Inputs:
  - x: Input data for this timestep, of shape (N, D).
  - prev_h: Hidden state from previous timestep, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)

  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - cache: Tuple of values needed for the backward pass.
  """
  next_h, cache = None, {}
  ##############################################################################
  # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
  # hidden state and any values you need for the backward pass in the next_h   #
  # and cache variables respectively.                                          #
  ##############################################################################

  # NxH.HxH
  m1 = prev_h.dot(Wh)
  # NxD.DxH
  m2 = m1 + x.dot(Wx)
  m3 = m2 + b
  next_h = np.tanh(m3)

  cache['x'] = x
  cache['prev_h'] = prev_h
  cache['Wx'] = Wx
  cache['Wh'] = Wh
  cache['m1'] = m1
  cache['m2'] = m2
  cache['m3'] = m3
  cache['next_h'] = next_h

  
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return next_h, cache


def rnn_step_backward(dnext_h, cache):
  """
  Backward pass for a single timestep of a vanilla RNN.
  
  Inputs:
  - dnext_h: Gradient of loss with respect to next hidden state (N, H)
  - cache: Cache object from the forward pass
  
  Returns a tuple of:
  - dx: Gradients of input data, of shape (N, D)
  - dprev_h: Gradients of previous hidden state, of shape (N, H)
  - dWx: Gradients of input-to-hidden weights, of shape (D, H)
  - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
  - db: Gradients of bias vector, of shape (H,)
  """
  dx, dprev_h, dWx, dWh, db = None, None, None, None, None
  ##############################################################################
  # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
  #                                                                            #
  # HINT: For the tanh function, you can compute the local derivative in terms #
  # of the output value from tanh.                                             #
  ##############################################################################
  x = cache['x']
  prev_h = cache['prev_h']
  Wx = cache['Wx']
  Wh = cache['Wh']
  m1 = cache['m1']
  m2 = cache['m2']
  m3 = cache['m3']
  next_h = cache['next_h']

  # next_h = np.tanh(m3)
  dm3 = (1 - np.tanh(m3)**2) * dnext_h

  # m3 = m2 + b
  dm2 = (1) * dm3
  db = (1) * np.sum(dm3, axis=0)
  
  # (N, H) <= x(N, D) . Wx(D, H)
  # m2 = m1 + x.dot(Wx)
  dm1 = (1) * dm2
  dWx = x.T.dot(dm2)     # (D, H) = x(N, D).T dm2(N, H)
  dx = dm2.dot(Wx.T)     # (N, D) = dm2(N, H) Wx(D, H).T
    
  # m1 = prev_h.dot(Wh)
  # dprev_h(N, H) = dm1(N, H) . Wh(H, H).T
  dprev_h = dm1.dot(Wh.T)
  # dWh(H, H) =  prev_h(N, H).T . dm1(N, H)
  dWh = prev_h.T.dot(dm1)


  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
  """
  Run a vanilla RNN forward on an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The RNN uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the RNN forward, we return the hidden states for all timesteps.
  
  Inputs:
  - x: Input data for the entire timeseries, of shape (N, T, D).
  - h0: Initial hidden state, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)
  
  Returns a tuple of:
  - h: Hidden states for the entire timeseries, of shape (N, T, H).
  - cache: Values needed in the backward pass
  """
  h, cache = None, {}
  ##############################################################################
  # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
  # input data. You should use the rnn_step_forward function that you defined  #
  # above.                                                                     #
  ##############################################################################
  N, T, D = x.shape
  H = h0.shape[1]

  prev_h = h0
  h = []
  h_cache = []
    
  for tidx in xrange(0, T):
    curr_x = x[:, tidx, :]
    next_h, next_h_cache = rnn_step_forward(curr_x, prev_h, Wx, Wh, b)
    h.append(next_h)
    h_cache.append(next_h_cache)
    prev_h = next_h

  h = np.array(h).transpose((1, 0, 2))
  cache['h_cache'] = h_cache
    
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return h, cache


def rnn_backward(dh, cache):
  """
  Compute the backward pass for a vanilla RNN over an entire sequence of data.
  
  Inputs:
  - dh: Upstream gradients of all hidden states, of shape (N, T, H)
  
  Returns a tuple of:
  - dx: Gradient of inputs, of shape (N, T, D)
  - dh0: Gradient of initial hidden state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
  - db: Gradient of biases, of shape (H,)
  """
  dx, dh0, dWx, dWh, db = None, None, None, None, None
  ##############################################################################
  # TODO: Implement the backward pass for a vanilla RNN running an entire      #
  # sequence of data. You should use the rnn_step_backward function that you   #
  # defined above.                                                             #
  ##############################################################################  
  h_cache = cache['h_cache']
  N, T, H = dh.shape
  D = h_cache[0]['x'].shape[1]
  
  dx = np.zeros((N, T, D))
  dh0 = np.zeros((N, H))
  dWx = np.zeros((D, H))
  dWh = np.zeros((H, H))
  db = np.zeros((H))
    
  #print dh.shape

  # https://www.reddit.com/r/cs231n/comments/473y6q/question_about_rnn_backward_assignment_3/
  # dh0 is the actual propogated gradient from dh through the RNN.
  # The rest of the dHs that are recorded would then need to be added to the running dH.
  for tidx in reversed(xrange(0, T)):
    # need to add dh0 from previous step to current dh
    cdx, cdprev_h, cdWx, cdWh, cdb = rnn_step_backward(dh[:, tidx, :] + dh0, h_cache[tidx])
    dx[:, tidx, :] = cdx
    dh0 = cdprev_h
    dWx += cdWx
    dWh += cdWh
    db += cdb
    
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
  """
  Forward pass for word embeddings. We operate on minibatches of size N where
  each sequence has length T. We assume a vocabulary of V words, assigning each
  to a vector of dimension D.
  
  Inputs:
  - x: Integer array of shape (N, T) giving indices of words. Each element idx
    of x muxt be in the range 0 <= idx < V.
  - W: Weight matrix of shape (V, D) giving word vectors for all words.
  
  Returns a tuple of:
  - out: Array of shape (N, T, D) giving word vectors for all input words.
  - cache: Values needed for the backward pass
  """
  out, cache = None, {}
  ##############################################################################
  # TODO: Implement the forward pass for word embeddings.                      #
  #                                                                            #
  # HINT: This should be very simple.                                          #
  ##############################################################################

  # https://www.reddit.com/r/cs231n/comments/48ndpi/problem_in_understanding_of_word_embedding/
  out = W[x, :]
  cache['W'] = W
  cache['x'] = x
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return out, cache


def word_embedding_backward(dout, cache):
  """
  Backward pass for word embeddings. We cannot back-propagate into the words
  since they are integers, so we only return gradient for the word embedding
  matrix.
  
  HINT: Look up the function np.add.at
  
  Inputs:
  - dout: Upstream gradients of shape (N, T, D)
  - cache: Values from the forward pass
  
  Returns:
  - dW: Gradient of word embedding matrix, of shape (V, D).
  """

  ##############################################################################
  # TODO: Implement the backward pass for word embeddings.                     #
  #                                                                            #
  # HINT: Look up the function np.add.at                                       #
  ##############################################################################
  x = cache['x']
  W = cache['W']
  V = W.shape[0]
  D = dout.shape[2]
  dW = np.zeros((V, D))

  # https://www.reddit.com/r/cs231n/comments/48ndpi/problem_in_understanding_of_word_embedding/
  np.add.at(dW, x, dout)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dW


def sigmoid(x):
  """
  A numerically stable version of the logistic sigmoid function.
  """
  pos_mask = (x >= 0)
  neg_mask = (x < 0)
  z = np.zeros_like(x)
  z[pos_mask] = np.exp(-x[pos_mask])
  z[neg_mask] = np.exp(x[neg_mask])
  top = np.ones_like(x)
  top[neg_mask] = z[neg_mask]
  return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
  """
  Forward pass for a single timestep of an LSTM.
  
  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.
  
  Inputs:
  - x: Input data, of shape (N, D)
  - prev_h: Previous hidden state, of shape (N, H)
  - prev_c: previous cell state, of shape (N, H)
  - Wx: Input-to-hidden weights, of shape (D, 4H)
  - Wh: Hidden-to-hidden weights, of shape (H, 4H)
  - b: Biases, of shape (4H,)
  
  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - next_c: Next cell state, of shape (N, H)
  - cache: Tuple of values needed for backward pass.
  """
  next_h, next_c, cache = None, None, {}
  #############################################################################
  # TODO: Implement the forward pass for a single timestep of an LSTM.        #
  # You may want to use the numerically stable sigmoid implementation above.  #
  #############################################################################

  # m0[N,4H] = x[N,D] . Wx[D,4H]
  m0 = x.dot(Wx)

  # m1[N,4H] = m0[N,4H] + h0[N,H] . Wh[H,4H]
  m1 = m0 + prev_h.dot(Wh)

  # act[N,4H] = m1[N,4H] + b[4H]
  act = m1 + b  
  ai, af, ao, ag = np.array_split(act, 4, axis = 1)

  i = sigmoid(ai)
  f = sigmoid(af)
  o = sigmoid(ao)
  g = np.tanh(ag)

  # m2[N,H] = f[N,H] * prev_c[N,H]
  m2 = f * prev_c

  # next_c[N,H] = m2[N,H] + i[N,H] * g[N,H]
  next_c = m2 + i * g

  # m3[N,H] = tanh(next_c[N,H])
  m3 = np.tanh(next_c)

  # next_h[N,H] = o[N,H] * m3[N,H]
  next_h = o * m3

  cache = (x, prev_h, Wx, Wh, ai, af, ao, ag, i, f, o, g, prev_c, next_c)


  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  
  return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
  """
  Backward pass for a single timestep of an LSTM.
  
  Inputs:
  - dnext_h: Gradients of next hidden state, of shape (N, H)
  - dnext_c: Gradients of next cell state, of shape (N, H)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient of input data, of shape (N, D)
  - dprev_h: Gradient of previous hidden state, of shape (N, H)
  - dprev_c: Gradient of previous cell state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  dx, dh, dc, dWx, dWh, db = None, None, None, None, None, None
  #############################################################################
  # TODO: Implement the backward pass for a single timestep of an LSTM.       #
  #                                                                           #
  # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
  # the output value from the nonlinearity.                                   #
  #############################################################################
  x, prev_h, Wx, Wh, ai, af, ao, ag, i, f, o, g, prev_c, next_c = cache

  do = np.tanh(next_c) * dnext_h
  dnext_c += o * (1 - np.tanh(next_c)**2) * dnext_h

  df = prev_c * dnext_c
  dprev_c = f * dnext_c
  di = g * dnext_c
  dg = i * dnext_c

  dai = di * sigmoid(ai) * (1 - sigmoid(ai))
  daf = df * sigmoid(af) * (1 - sigmoid(af))
  dao = do * sigmoid(ao) * (1 - sigmoid(ao))
  dag = dg * (1 - np.tanh(ag)**2)

  dact = np.concatenate((dai, daf, dao, dag), axis = 1)

  dm1 = dact * 1
  db = np.sum(dact, axis = 0) #

  dm0 = dm1 * 1
  dprev_h = dm1.dot(Wh.T) #
  dWh = prev_h.T.dot(dm1) #
  
  dx = dm0.dot(Wx.T) #
  dWx = x.T.dot(dm0) #
  
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
  """
  Forward pass for an LSTM over an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the LSTM forward, we return the hidden states for all timesteps.
  
  Note that the initial cell state is passed as input, but the initial cell
  state is set to zero. Also note that the cell state is not returned; it is
  an internal variable to the LSTM and is not accessed from outside.
  
  Inputs:
  - x: Input data of shape (N, T, D)
  - h0: Initial hidden state of shape (N, H)
  - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
  - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
  - b: Biases of shape (4H,)
  
  Returns a tuple of:
  - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
  - cache: Values needed for the backward pass.
  """

  #############################################################################
  # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
  # You should use the lstm_step_forward function that you just defined.      #
  #############################################################################
  N, T, D = x.shape
  _, H = h0.shape
  h, cache = np.zeros((T, N, H)), []

  prev_h = h0
  prev_c = np.zeros((N, H))
  
  for step in range(0, T):
    xt = x[:, step, :] 
    next_h, next_c, next_cache = lstm_step_forward(xt, prev_h, prev_c, Wx, Wh, b)
   
    h[step] = next_h
    prev_h = next_h
    prev_c = next_c
    
    cache.append(next_cache)
 
  h = h.transpose(1, 0, 2)

  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  
  return h, cache


def lstm_backward(dh, cache):
  """
  Backward pass for an LSTM over an entire sequence of data.]
  
  Inputs:
  - dh: Upstream gradients of hidden states, of shape (N, T, H)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient of input data of shape (N, T, D)
  - dh0: Gradient of initial hidden state of shape (N, H)
  - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  dx, dh0, dWx, dWh, db = None, None, None, None, None
  #############################################################################
  # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
  # You should use the lstm_step_backward function that you just defined.     #
  #############################################################################
  x, _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _ = cache[0]
  N, T, H = dh.shape
  D = x.shape[-1]

  dh = dh.transpose(1, 0, 2)
  tdprev_h = np.zeros((N, H))
  tdprev_c = np.zeros((N, H))

  dx = np.zeros((T, N, D))
  dh0 = np.zeros((N, H))
  dWx = np.zeros((D, 4*H))
  dWh = np.zeros((H, 4*H))
  db = np.zeros((4 * H,))
    
  for step in reversed(xrange(T)):
    dcurr_h = dh[step] + tdprev_h
    dcurr_c = tdprev_c
    
    tdx, tdprev_h, tdprev_c, tdWx, tdWh, tdb = lstm_step_backward(dcurr_h, dcurr_c, cache[step])
    
    dx[step] = tdx
    dh0 = tdprev_h

    # since it's the same weights across all time steps in each LSTM, need to accumulate gradients
    dWx += tdWx
    dWh += tdWh
    db += tdb
  
  dx = dx.transpose(1, 0, 2)
  # only wanted the very last one.
  
    
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  
  return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
  """
  Forward pass for a temporal affine layer. The input is a set of D-dimensional
  vectors arranged into a minibatch of N timeseries, each of length T. We use
  an affine function to transform each of those vectors into a new vector of
  dimension M.

  Inputs:
  - x: Input data of shape (N, T, D)
  - w: Weights of shape (D, M)
  - b: Biases of shape (M,)
  
  Returns a tuple of:
  - out: Output data of shape (N, T, M)
  - cache: Values needed for the backward pass
  """
  N, T, D = x.shape
  M = b.shape[0]
  out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
  cache = x, w, b, out
  return out, cache


def temporal_affine_backward(dout, cache):
  """
  Backward pass for temporal affine layer.

  Input:
  - dout: Upstream gradients of shape (N, T, M)
  - cache: Values from forward pass

  Returns a tuple of:
  - dx: Gradient of input, of shape (N, T, D)
  - dw: Gradient of weights, of shape (D, M)
  - db: Gradient of biases, of shape (M,)
  """
  x, w, b, out = cache
  N, T, D = x.shape
  M = b.shape[0]

  dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
  dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
  db = dout.sum(axis=(0, 1))

  return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
  """
  A temporal version of softmax loss for use in RNNs. We assume that we are
  making predictions over a vocabulary of size V for each timestep of a
  timeseries of length T, over a minibatch of size N. The input x gives scores
  for all vocabulary elements at all timesteps, and y gives the indices of the
  ground-truth element at each timestep. We use a cross-entropy loss at each
  timestep, summing the loss over all timesteps and averaging across the
  minibatch.

  As an additional complication, we may want to ignore the model output at some
  timesteps, since sequences of different length may have been combined into a
  minibatch and padded with NULL tokens. The optional mask argument tells us
  which elements should contribute to the loss.

  Inputs:
  - x: Input scores, of shape (N, T, V)
  - y: Ground-truth indices, of shape (N, T) where each element is in the range
       0 <= y[i, t] < V
  - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
    the scores at x[i, t] should contribute to the loss.

  Returns a tuple of:
  - loss: Scalar giving loss
  - dx: Gradient of loss with respect to scores x.
  """

  N, T, V = x.shape
  
  x_flat = x.reshape(N * T, V)
  y_flat = y.reshape(N * T)
  mask_flat = mask.reshape(N * T)
  
  probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
  dx_flat = probs.copy()
  dx_flat[np.arange(N * T), y_flat] -= 1
  dx_flat /= N
  dx_flat *= mask_flat[:, None]
  
  if verbose: print 'dx_flat: ', dx_flat.shape
  
  dx = dx_flat.reshape(N, T, V)
  
  return loss, dx

