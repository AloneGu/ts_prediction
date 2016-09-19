#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 9/19/16 2:21 PM
# @Author  : Jackling 

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from util import make_data


class GBRTPredictor(object):
    """
    use gbrt to do the prediction
    """

    def __init__(self, input_labels, output_labels, input_len, output_len):
        """
        :param df: data frame
        :param input_labels: list of str,which labels to use as x
        :param output_labels: list of str,which lables to predict
        :param input_len: int, x length
        :param output_len: int, y length
        :return: return self obj
        """
        self.input_labels = input_labels
        self.output_labels = output_labels
        self.input_len = input_len
        self.output_len = output_len
        self.models = []  # label : [model,model]  like 'a':[model_1,model_2]

        # init models
        self.model_cnt = self.output_len * len(self.output_labels)
        self.models = [GradientBoostingRegressor() for i in range(self.model_cnt)]

    def fit(self, df):
        # prepare x_data and y_data for training
        x = df[self.input_labels].values
        y = df[self.output_labels].values
        x_data, y_data = make_data(x, y, self.input_len, self.output_len)

        # fit each model
        for i in range(self.model_cnt):
            tmp_y = y_data[:, i]
            self.models[i].fit(x_data, tmp_y)

    def predict(self, df):
        """
        :param df: input dataframe, if df len is larger than input_len, only return last prediction result
        :return:
        """
        x = df[self.input_labels].values[-self.input_len:]

        # reshape x for training
        x_data = x.reshape(-1, self.input_len * len(self.input_labels))

        # use each model to predict the result
        y = np.array([self.models[i].predict(x_data) for i in range(self.model_cnt)])

        # reshape y for return
        y_data = y.reshape(-1, len(self.output_labels))
        return y_data
