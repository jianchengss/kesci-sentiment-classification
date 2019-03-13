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


word_max_length = 10
tfidf_min_df = 2