#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 9/19/16 4:42 PM
# @Author  : Jackling 


from __future__ import print_function
from garch_predictor import NeuralGARCH


class NeuralARCH(NeuralGARCH):
    ''' ARCH(q) is GARCH(0, q) '''

    def __init__(self, q, loss='mean_squared_error', optimizer='sgd'):
        super(NeuralARCH, self).__init__(p=1, q=q, loss=loss, optimizer=optimizer)
