#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 9/19/16 3:52 PM
# @Author  : Jackling 


DATA_FILE = '../data/nyc_taxi.csv'

import pandas as pd

df = pd.read_csv(DATA_FILE)[:480]

print len(df)

print df.head()

from ts_predictor import gbrt_predictor

model = gbrt_predictor.GBRTPredictor(['value', 'hour', 'sqrt_value'], ['value'], 48, 4)

model.fit(df)
print 'fit done'

res = model.predict(df[10:58])
print 'predict done'

print 'pred', res.flatten().round()
print 'truth', df['value'][58:62].values
