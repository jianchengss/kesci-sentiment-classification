#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-3-7 下午4:16
# @Author  : J.W.
# @File    : random-baseline.py


import pandas as pd
import random

path = "./data/test.csv"
# df_result = pd.read_csv(path, escapechar=)
# print(df_result.head())
import codecs

random.seed(1)
fr = codecs.open(path, 'rb', 'utf-8', 'ignore')
df_result = pd.read_csv(fr, lineterminator='\n')
print(df_result.head())
print(df_result.shape)
data = []
ids = df_result["ID"]
for id in ids:
    try:
        float(id)
    except:
        continue
    r = random.random()
    # print("%s,    %s" % (str(id), str(r)))
    data.append([id, r])
result = pd.DataFrame(data, columns=['ID', "Pred"])
print(result.head())
result.to_csv("./data/random-baseline.csv", index=0)
print(df_result.shape)
result_ids = list(result["ID"])

print("=======")
for i in result_ids:
    if i not in list(ids):
        print(i)
