#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-3-8 下午4:33
# @Author  : J.W.
# @File    : predict.py


import pandas as pd

import comm
from feature import Feature

# 使用的模型文件
model_path = "./submissions/model_neural_clf_0.8352"
reslut_path = model_path.replace('model', 'reslut').replace('0.', '')
reslut_path = reslut_path + ".csv"  # 结果保存路径
model = comm.load_file(model_path)  # 加载模型

feature = Feature()
test_feature = feature.test_X  # 测试文件特征
ids = feature.test_ids
# 预测过程，结果为两个label的概率
predict_proba = model.predict_proba(test_feature)
proba = predict_proba[:, 1]  # 这里只取1的概率

data = []  # 最终结果
assert len(ids) == len(proba)
for id, p in zip(ids, proba):
    data.append([id, p])
result = pd.DataFrame(data, columns=['ID', "Pred"])
comm.dump_submission(result, path=reslut_path)
