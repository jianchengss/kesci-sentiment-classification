#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-3-8 下午3:01
# @Author  : J.W.
# @File    : config.py
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, 'data')
resource_path = os.path.join(BASE_DIR, 'resource')

LOG_DIR = os.path.join(BASE_DIR, "logs")

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)  # 创建路径

log_file = os.path.join(LOG_DIR, "app.log")

DEBUG = False
DEBUG = True

# path
train_data_path = "./data/train.csv"
test_data_path = "./data/test.csv"
submission_path = "./data/submission.csv"

model_path = './data/model'

# 特征
accept_columns = []
accept_columns.append('job')
accept_columns.append('marital')
accept_columns.append('education')
accept_columns.append('housing')
accept_columns.append('loan')  # 个人贷款
accept_columns.append('contact')  # 联系方式类型
accept_columns.append('day')
accept_columns.append('month')
accept_columns.append('campaign')  # 本次活动中，与该客户交流过的次数
accept_columns.append('pdays')  # 距离上次活动最后一次联系该客户，过去了多久（999表示没有联系过）
accept_columns.append('previous')  # 本次活动之前，与该客户交流过的次数
accept_columns.append('age')
accept_columns.append('default')  # 是否有违约记录
# accept_columns.append('poutcome')
accept_columns.append('duration')
# accept_columns.append('balance')
# accept_columns.append('previous')  # 在本次活动之前，与该客户交流过的次数

# 如果用到这些特征 需要编码
all_need_coder_columns = ['job', 'marital', 'education', 'housing',
                          'default', 'loan', 'contact', 'poutcome', 'month']

# 有无的特征
extention_features = []

# 阈值特征 大于指定值为0  小于等于为1
thrshold_columns = {}
thrshold_columns['age'] = 30
thrshold_columns['pdays'] = 200
thrshold_columns['campaign'] = 3
thrshold_columns['previous'] = 40
thrshold_columns['balance'] = 0
thrshold_columns['duration'] = 80

# one_hot_extension_features
# 年龄 30 50 80 分几档
one_hot_extension_features = []
one_hot_extension_features.append('age')
