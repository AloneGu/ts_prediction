#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 9/19/16 4:42 PM
# @Author  : Jackling 


from __future__ import print_function

from keras.models import Sequential
from keras.layers.core import Dense

from ts_predictor.data_generator import train_test_split
from ts_predictor.layers.recurrent import ARMA, GARCH
from ts_predictor.preprocessing.reshape import sliding_window

import matplotlib.pyplot as plt
import numpy as np

from garch_predictor import NeuralGARCH


class NeuralARCH(NeuralGARCH):
    ''' ARCH(q) is GARCH(0, q) '''

    def __init__(self, q, loss='mean_squared_error', optimizer='sgd'):
        super(NeuralARCH, self).__init__(p=1, q=q, loss=loss, optimizer=optimizer)
