#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 13:10:42 2017

@author: danny
"""
import dynet as dy
import speechEncoder
import numpy as np

filter_height = [40, 1, 1]
filter_width = [5, 25, 25]
channels = [1, 64, 512]
num_filters = [64, 512, 1024]
model = dy.ParameterCollection()
stride = [1, 1, 1]

dy.renew_cg()

x = speechEncoder.speechBuilder(filter_height, filter_width, channels, num_filters, model, stride)

src = dy.inputTensor(np.random.randint(1,100,(40,1000,1,5)), batched = True)

out = x.transduce(src)

out = out.npvalue()