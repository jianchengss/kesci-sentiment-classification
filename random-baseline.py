#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-3-7 下午4:16
# @Author  : J.W.
# @File    : random-baseline.py


import pandas as pd
import random
import codecs

random.seed(1)

test_file_path = "./data/test.csv"
result_path = "./submissions/random-baseline.csv"

fr = codecs.open(test_file_path, 'rb', 'utf-8', 'ignore')
test_df = pd.read_csv(fr, lineterminator='\n')
print(test_df.head())
print(test_df.shape)
print(test_df.columns)
data = []
ids = test_df["ID"]
review = test_df['review']
# for id , text in zip(ids, review):
#     print("{} {}".format(id,text))
for id in ids:
    r = random.random()
    # print("%d,    %f" % (id, r))
    data.append([id, r])
result = pd.DataFrame(data, columns=['ID', "Pred"])
print(result.head())
print(result.shape)
result.to_csv(result_path, index=0)
