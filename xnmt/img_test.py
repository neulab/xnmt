#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 17:16:53 2017

@author: danny
"""

import dynet as dy
import vgg16Encoder
import numpy as np

reload(vgg16Encoder)
model = dy.ParameterCollection()
in_width = 4096
out_width = 1024

dy.renew_cg()

x = vgg16Encoder.vgg16Builder(in_width, out_width, model)

src = dy.inputTensor(np.random.randint(1,100,(4096,1,5)), batched = True)

out = x.transduce(src)

out = out.npvalue()