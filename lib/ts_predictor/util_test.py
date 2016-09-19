#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 9/19/16 3:23 PM
# @Author  : Jackling 

import numpy as np
from util import *


def test_make_data():
    x = np.arange(40).reshape(10, 4)
    y = x[:, :2]
    x_data, y_data = make_data(x, y, 5, 3)
    for a, b in zip(x_data, y_data):
        print a, b


test_make_data()
