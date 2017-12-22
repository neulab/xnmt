import numpy as np
import dynet as dy


def LeCunUniform(fan_in, scale=1.0):
  """
  Reference: LeCun 98, Efficient Backprop
  http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
  """
  s = scale * np.sqrt(3. / fan_in)
  return dy.UniformInitializer(s)
