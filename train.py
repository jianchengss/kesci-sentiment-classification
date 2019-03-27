#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-11-19 上午10:22
# @Author  : J.W.
# @File    : train.py


from collections import Counter

import numpy as np
import random
# import xgboost as xgb
from sklearn import metrics, svm
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

import config
from logger import logger
from report import Report

classfiers = {}

features_name = []

C = 1.0
# classfiers['lc'] = linear_model.SGDClassifier(max_iter=100, tol=1e-3, early_stopping=True, random_state=10 ) # random_state=10,
# classfiers['lr'] = LogisticRegression(random_state=0,C=10, penalty='l2', solver='saga', multi_class='ovr')
# classfiers['xgb-1'] = xgb.XGBClassifier(max_depth=3, n_estimators=400, learning_rate=0.05)
# classfiers['xgb-2'] = xgb.XGBClassifier(max_depth=3, n_estimators=400, learning_rate=0.01)
# classfiers['xgb-3'] = xgb.XGBClassifier(max_depth=3, n_estimators=200, learning_rate=0.05)
# classfiers['xgb-4'] = xgb.XGBClassifier(max_depth=6, n_estimators=400, learning_rate=0.05)
# xgb_5 = xgb.XGBClassifier(
#     max_depth=4,  # 增加树的深度
#     learning_rate=0.05,  # 减小学习率
#     n_estimators=500,  # 增加分类器的个数
#     n_jobs=2,
#     random_state=22,
#     seed=11
# )

# classfiers['xgb-5']  = xgb_5
svm_1 = svm.SVC(C=1.0, kernel='rbf', gamma='auto', probability=True)
rfc_1 = RandomForestClassifier(n_estimators=100, random_state=10)
rfc_2 = RandomForestClassifier(n_estimators=200, random_state=10)
rfc_3 = RandomForestClassifier(n_estimators=300, random_state=10)
rfc_4 = RandomForestClassifier(n_estimators=200, max_depth=13, min_samples_split=80, min_samples_leaf=10,
                               oob_score=True, random_state=10, max_features='sqrt')
rfc_5 = RandomForestClassifier(random_state=30, n_estimators=30, max_depth=11, max_features=0.5,
                               criterion='entropy', min_samples_split=140, min_samples_leaf=50)
# Build a forest and compute the feature importances
forest_1 = ExtraTreesClassifier(n_estimators=25, random_state=0)
forest_2 = ExtraTreesClassifier(n_estimators=250, random_state=0)
rng = np.random.RandomState(1)
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),
                         algorithm="SAMME",
                         n_estimators=100)

# Decision Tree
tree_clf = tree.DecisionTreeClassifier()

# Gradient Boosting Classifier
grad_clf = GradientBoostingClassifier()

# Random Forest Classifier
rand_clf = RandomForestClassifier(n_estimators=18)

# NeuralNet Classifier
neural_clf = MLPClassifier(alpha=1)

# Naives Bayes
nav_clf = GaussianNB()

voting_clf = VotingClassifier(
    estimators=[('gbc', grad_clf), ('nav', nav_clf), ('neural', neural_clf)],
    voting='soft'
)
voting_rfc = VotingClassifier(
    estimators=[('rfc', rfc_1), ('rfc_2', rfc_2), ('rfc_3', rfc_3)],
    voting='soft')
# classfiers['rfc-1'] = rfc_1
# classfiers['rfc-2'] = rfc_2
# classfiers['rfc-3'] = rfc_3
# classfiers['rfc_4'] = rfc_4
# classfiers['rfc_5'] = rfc_5
#
classfiers['neural_clf'] = neural_clf

classfiers['nav_clf'] = nav_clf
classfiers['grad_clf'] = grad_clf
classfiers['tree_clf'] = tree_clf
classfiers['forest-1'] = forest_1
classfiers['forest-2'] = forest_2
classfiers['voting_clf'] = voting_clf
classfiers['voting_rfc'] = voting_rfc


# classfiers['svm-svc-2'] = svm.SVC(C=2.0, kernel='rbf', gamma='auto')


def random_result(num=0):
    '''
    随机生成结果，生成baseline
    :param num:
    :return:
    '''
    result = []
    for i in range(num):
        result.append(random.randint(1, 2))
    return result


def cv(X, y, clf):
    '''
    交叉验证 Cross-validation
    :param X: 训练集特征
    :param y: 目标label
    :param clf: 分类器
    :return: 交叉验证过程中的最佳模型
    '''
    model = None
    max_p = 0  # 记录实验中最好的模型
    reporter = Report('train')  # 定义评价器
    for i in range(1, 10):  # 10
        logger.info("Folder {}".format(i))
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21 + i)
        if config.DEBUG:
            logger.info(
                "Train x data: ({}, {}), Train y data: {}".format(len(x_train), len(x_train[0]), Counter(y_train)))
            logger.info("Test x data: ({}, {}), Test y data: {}".format(len(x_test), len(x_test[0]), Counter(y_test)))

        clf.fit(x_train, y_train)  # 训练过程
        predict_proba = clf.predict_proba(x_test)  # 返回每个类别的概率值

        # TODO 随机 看结果对比submission结果
        p = reporter.report_one_folder(y_test, predict_proba, threshold=0.5)
        if p > max_p:
            max_p = p
            logger.info("Max result: {:.4f}".format(p))
            model = clf
    reporter.report_final_result()
    return model, reporter


def random_forest_param_search(X, y):
    '''
    调参过程
    https://blog.csdn.net/yingfengfeixiang/article/details/79369059
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    :param x:
    :param y:
    :return:
    '''
    from sklearn.model_selection import GridSearchCV
    # 查看基准的训练情况
    logger.info("base")
    rfo = RandomForestClassifier(oob_score=True, random_state=10)
    rfo.fit(X, y)
    logger.info("oob_score: {}".format(rfo.oob_score_))
    y_predprob = rfo.predict_proba(X)[:, 1]
    logger.info("AUC Score (Train): {}".format(metrics.roc_auc_score(y, y_predprob)))

    # 把调参的结果一步一步加入进来看对比效果
    logger.info("para:")
    rfo = RandomForestClassifier(n_estimators=185, random_state=10, max_features=0.5, max_depth=7, oob_score=True)
    rfo.fit(X, y)
    logger.info("oob_score: {}".format(rfo.oob_score_))
    y_predprob = rfo.predict_proba(X)[:, 1]
    logger.info("AUC Score (Train): {}".format(metrics.roc_auc_score(y, y_predprob)))

    # 一步一步的搜索最佳参数，调后面参数的时候把前面参数带入
    param_test = {}
    param_test = {'n_estimators': range(170, 200, 5)}  # 185
    # param_test = {'max_depth': range(3, 20, 2)}  # 7
    # param_test = {'max_features': ['auto', 'sqrt', 'log2', .01, .5, .99]}  # .5
    # param_test = {'criterion': ['gini', 'entropy']}  # gini
    # param_test = {'splitter': ['best', 'random']} # error
    # param_test = {'min_samples_split': range(80, 150, 20), 'min_samples_leaf': range(10, 60, 10)}  # 140, 50
    # param_test = {'n_estimators': range(100, 3000, 500),
    #               'max_depth': range(3, 20, 2),
    #               'max_features': ['auto', 'sqrt', 'log2', .01, .5, .99],
    #               'criterion': ['gini', 'entropy'],
    #               'min_samples_split': range(80, 150, 20),
    #               'min_samples_leaf': range(10, 60, 10)}

    logger.info("param test: {}".format(param_test))
    gsearch1 = GridSearchCV(
        estimator=RandomForestClassifier(random_state=10),
        param_grid=param_test, scoring='roc_auc', cv=5)
    gsearch1.fit(X, y)
    logger.info("best_estimator_ {} ".format(gsearch1.best_estimator_))
    logger.info("best para: {}".format(gsearch1.best_params_))


# def predict(model, feature):
#     result = model.predict(feature)
#     result_pro = model.predict_proba(feature)
#     # logger.info("result: {}".format(result))
#     return result, result_pro

def train(X, y, clf):
    clf.fit(X, y)
    # logger.info("oob_score: {}".format(clf.oob_score_))
    y_predprob = clf.predict_proba(X)[:, 1]
    logger.info("AUC Score (Train): {}".format(metrics.roc_auc_score(y, y_predprob)))
    comm.save_file(clf, config.model_path)


def train_cv(X, y):
    '''
    对classfiers中的每个分类器执行 CV过程
    '''
    reporters = {}
    for index, (name, clf) in enumerate(classfiers.items()):
        logger.info('{}: {}'.format(name, clf))
        model, train_reporter = cv(X, y, clf)
        model_path = config.model_path + '_' + name + "_{:.4f}".format(train_reporter.auc)

        comm.save_file(model, model_path)
        reporters[name] = [train_reporter]
        # score = cross_val_score(clf, X, y, cv=10)
        # logger.info("cross_val_score: {} ".format(score.mean()))

    logger.info("Sum result:\n")
    logger.info('===========')
    for name, train_reporter in reporters.items():
        logger.info('-----')
        logger.info("clf: {}".format(name))
        train_reporter[0].report_final_result()


if __name__ == "__main__":
    import comm
    from feature import Feature

    f = Feature()  # 特征工程

    # random_forest_param_search(X, y)
    train_cv(f.X, f.y)  # 十折交叉过程
    # train(X, y, voting_clf)
