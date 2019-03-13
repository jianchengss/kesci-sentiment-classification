#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-3-8 下午4:33
# @Author  : J.W.
# @File    : predict.py


import pandas as pd
import random

import comm
from feature import Feature

model_path = "./submissions/model_forest-2_0.8269"
reslut_path = model_path.replace('model', 'reslut').replace('0.', '')
reslut_path = reslut_path + ".csv"
model = comm.load_file(model_path)

feature = Feature()
test_feature = feature.test_X

ids = feature.test_ids
predict_proba = model.predict_proba(test_feature)
proba = predict_proba[:, 1]

data = []

# idx = 1
# diff = 1  # 记录偏移量  id中没有 355
# while (idx < 2713):
#     if ids[idx - diff] != idx:
#         r = float("{:.6f}".format(random.random()))
#         data.append([idx, r])
#         idx += 1
#         diff += 1
#     else:
#         data.append([idx, proba[idx - diff]])
#         idx += 1
assert len(ids) == len(proba)
for id , p in zip(ids, proba):
    data.append([id, p])

result = pd.DataFrame(data, columns=['ID', "Pred"])
comm.dump_submission(result, path=reslut_path)
