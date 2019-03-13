#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-11-30 下午2:39
# @Author  : J.W.
# @File    : report.py

from collections import Counter

import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score

from logger import logger


def view_predict(y, y_hat):
    '''
    预测值的分布情况
    :param y:
    :param y_hat:
    :return:
    '''
    y_counter = Counter(y)
    y_hat_counter = Counter(y_hat)
    logger.info("y/y_hat stat: {} / {}".format(y_counter, y_hat_counter))


class Report():
    '''
    对预测结果进行评估
    '''

    def __init__(self, name=''):
        '''
        初始化
        '''
        self.name = name
        self.f1 = []
        self.p1 = []
        self.r1 = []

        self.f0 = []
        self.p0 = []
        self.r0 = []
        self.auc_scores = []
        self.auc = 0

    def report_one_folder_by_lable(self, y, y_hat):
        '''
        中间结果
        '''

        # view_predict(y, y_hat)
        f1_1 = metrics.f1_score(y, y_hat, pos_label=1)
        p_1 = metrics.precision_score(y, y_hat, pos_label=1)
        r_1 = metrics.recall_score(y, y_hat, pos_label=1)

        f1_0 = metrics.f1_score(y, y_hat, pos_label=0)
        p_0 = metrics.precision_score(y, y_hat, pos_label=0)
        r_0 = metrics.recall_score(y, y_hat, pos_label=0)

        self.f1.append(f1_1)
        self.p1.append(p_1)
        self.r1.append(r_1)

        self.f0.append(f1_0)
        self.p0.append(p_0)
        self.r0.append(r_0)

        logger.info("0 result p: {:.4f} r: {:.4f} f {:.4f}".format(p_0, r_0, f1_0))
        logger.info("1 result p: {:.4f} r: {:.4f} f {:.4f}".format(p_1, r_1, f1_1))
        # logger.info("0 result p: {:.4f} r: {:.4f} f {:.4f} / 1 result p: {:.4f} r: {:.4f} f {:.4f}".
        #             format(p_0, r_0, f1_0, p_1, r_1, f1_1))
        return p_1

    def report_one_folder(self, y, predict_proba, threshold=0.5):
        '''
        中间结果
        '''
        logger.info(self.name)
        y_hat = [1 if score[1] >= threshold else 0 for score in predict_proba]
        p_1 = self.report_one_folder_by_lable(y, y_hat)

        # auc evaluate
        y_score = predict_proba[:, 1]
        auc = roc_auc_score(y, y_score, average="macro")
        self.auc_scores.append(auc)
        logger.info("AUC score: {:.4f}".format(auc))

        logger.info('============')

        self.auc = np.mean(self.auc_scores)
        return auc

    def report_final_result(self):
        '''
        最终结果
        '''
        logger.info(self.name)
        logger.info(
            "0 avg result p: {:.4f} r: {:.4f} f {:.4f}".format(np.mean(self.p0), np.mean(self.r0), np.mean(self.f0)))
        logger.info(
            "1 avg result p: {:.4f} r: {:.4f} f {:.4f}".format(np.mean(self.p1), np.mean(self.r1), np.mean(self.f1)))

        if len(self.auc_scores) > 0:
            logger.info(
                "AUC avg score: {:.4f}".format(np.mean(self.auc_scores)))
