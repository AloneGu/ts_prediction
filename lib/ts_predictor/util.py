#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 9/19/16 2:22 PM
# @Author  : Jackling 

import numpy as np


def make_data(x, y, input_len, output_len):
    total_len = len(x)
    total_combination_cnt = total_len - input_len - output_len + 1
    x_data = []
    y_data = []
    for i in range(total_combination_cnt):
        tmp_x = x[i:i + input_len]
        x_data.append(np.array(tmp_x).flatten())
        tmp_y = y[i + input_len:i + input_len + output_len]
        y_data.append(np.array(tmp_y).flatten())
    return np.array(x_data), np.array(y_data)
