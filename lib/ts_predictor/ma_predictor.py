#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 9/19/16 4:42 PM
# @Author  : Jackling 


from __future__ import print_function

from keras.models import Sequential
from keras.layers.core import Dense

from giz_predictor.data_generator import train_test_split
from giz_predictor.layers.recurrent import ARMA, GARCH
from giz_predictor.preprocessing.reshape import sliding_window

import matplotlib.pyplot as plt
import numpy as np

from ts_models import ForecastModel


class NeuralMA(ForecastModel):
    def __init__(self, q, loss='mean_squared_error', optimizer='sgd'):
        self.q = q
        super(NeuralMA, self).__init__()

        model = Sequential()
        model.add(ARMA(inner_input_dim=self.q, input_shape=(1, 1), output_dim=1,
                       activation='linear', ma_only=False))
        model.compile(loss=loss, optimizer=optimizer)
        self.model = model

    def preprocess(self, ts, train_percentage):
        if len(ts.shape) == 1:
            ts = ts.reshape(1, len(ts))

        X, y = sliding_window(ts, p=1, drop_last_dim=False)
        (X_train, y_train), (X_test, y_test) = train_test_split(X, y, train_percentage=train_percentage)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.has_data = True
