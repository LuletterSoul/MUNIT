#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software: 
@file: torch_test.py
@time: 6/17/20 11:12 AM
@version 1.0
@desc:
"""

import torch
import torch.nn.functional as F
from torch import multiprocessing

input = torch.randn((3, 2), requires_grad=True)
target = torch.rand((3, 2), requires_grad=False)
print(input)
print(target)


multiprocessing.Manager().list()
print(torch.sigmoid(input))
loss = F.binary_cross_entropy(torch.sigmoid(input), target)
print(loss)
loss.backward()

# input_tensor = torch.randn((3, 2), requires_grad=True)
# target = torch.rand((3, 2), requires_grad=False)
# print(F.sigmoid(input_tensor))
# F.cross_entropy(F.sigmoid(input_tensor), target)
