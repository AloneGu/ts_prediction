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

from ts_models import ForecastModel


class NeuralGARCH(ForecastModel):
    '''
    Starting from an input time-series ts, we fit an AR(q) model to it and compute the residuals.
    In a GARCH process we assume that volatility at time t is governed by an ARMA(p, q) process on
    p squared residuals up to time t-p and q previously computed squared vol terms. We use exactly this
    property to predict squared volatilities, i.e. to obtain actual sigmas one has to take square roots
    of the resulting time-series.

    The network is optimized by backprop, checking predicted vols against historic vols.

    TODO: Check validity of approach and assumptions.
    TODO: Properly take care of initial vol values.
    TODO: On which window do we actually compute historic vols?
    TODO: Think about some extensions (EGARCH, GJR, etc.)

    Parameters:
    -----------
    p: int, number of sigma terms in GARCH
    q: int, number of residual terms in GARCH
    '''

    def __init__(self, p, q, loss='mean_squared_error', optimizer='sgd'):
        if q <= 0:
            raise ValueError('q must be strictly positive')
        self.p = p
        self.q = q
        super(NeuralGARCH, self).__init__()

        # We regress the original time-series to the best-fitting AR(q) model to extract error terms
        self.regressor_model = NeuralAR(p=self.q)

        # Initialize the actual garch model that will be trained on squared residuals
        model = Sequential()
        model.add(GARCH(inner_input_dim=self.p, input_shape=(self.q, 1), output_dim=1, activation='linear'))
        model.compile(loss=loss, optimizer=optimizer)
        self.model = model

    def preprocess(self, ts, train_percentage):
        if len(ts.shape) == 1:
            ts = ts.reshape(1, len(ts))

        # Regress original ts
        X_orig, y_orig = sliding_window(ts, p=self.q, drop_last_dim=True)
        self.regressor_model.preprocess(ts, 1.0)
        self.regressor_model.fit(X_orig, y_orig)

        # Predict with regressor model and substract original to obtain residuals.
        pred = self.regressor_model.model.predict(X_orig)
        pred = pred.reshape(1, len(pred))
        residual_ts = np.concatenate((np.zeros((1, self.q)), pred), axis=1) - ts

        # Define sliding window on residuals and compute historic vols from it
        X_residual, y_residual = sliding_window(residual_ts, p=self.q, drop_last_dim=False)
        y_sigmas = X_residual.std(axis=1)
        y_sigmas_squared = y_sigmas * y_sigmas
        X_residual_squared = X_residual * X_residual

        # Compute train-test-split and set internal variables
        (X_train, y_train), (X_test, y_test) = train_test_split(X_residual_squared, y_sigmas_squared,
                                                                train_percentage=train_percentage)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.has_data = True
