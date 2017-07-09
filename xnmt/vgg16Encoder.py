#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 13:38:18 2017

@author: danny
"""

import math
import dynet as dy
from residual import PseudoState
from embedder import ExpressionSequence

class vgg16Builder(object):
  """
  Inputs are first put through 2 CNN layers, each with stride (2,2), so dimensionality
  is reduced by 4 in both directions.
  Then, we add a configurable number of bidirectional RNN layers on top.
  """
  
  def __init__(self, in_height, out_height, model):
    """
    :param num_layers: depth of the RNN
    :param input_dim: size of the inputs
    :param hidden_dim: size of the outputs (and intermediate RNN layer representations)
    :param model
    :param rnn_builder_factory: RNNBuilder subclass, e.g. LSTMBuilder
    """
  
    
    self.in_height = in_height
    self.out_height = out_height
    
    normalInit=dy.NormalInitializer(0, 0.1)
    self.pW = model.add_parameters(dim = (self.out_height, self.in_height), init=normalInit)
    self.pb = model.add_parameters(dim = self.out_height)
  def whoami(self): return "vgg16Encoder"

  def transduce(self, src):
    #src = src.as_tensor()

    src_height = src.dim()[0][0]
    src_width = src.dim()[0][1]
    batch_size = src.dim()[1]
    
    W = dy.parameter(self.pW)
    b = dy.parameter(self.pb)
    
    src = dy.reshape(src, (src_height, src_width), batch_size=batch_size) # ((276, 80, 3), 1)
    # convolution and pooling layers    
    l1 = (W*src)+b
    output = dy.cdiv(l1,dy.sqrt(dy.squared_norm(l1)))
    return output

  def initial_state(self):
    return PseudoState(self)
