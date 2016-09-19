#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 9/19/16 3:06 PM
# @Author  : Jackling 

DATA_FILE = '../data/nyc_taxi.csv'

import pandas as pd

df = pd.read_csv(DATA_FILE)

print df.head()
