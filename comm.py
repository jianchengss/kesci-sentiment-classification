#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-6-21 下午9:31
# @Author  : J.W.
# @File    : comm.py

import codecs
import datetime
import pandas as pd
import pickle
from datetime import date

import config
from logger import logger


def calculate_age(born):
    '''
    出生日期转换成年龄
    :param born:  date类型
    :return:年龄
    '''
    if isinstance(born, datetime):
        born = date(born.year, born.month, born.day)

    if not isinstance(born, date):
        return -1

    today = date.today()
    try:
        birthday = born.replace(year=today.year)
    except ValueError:
        # raised when birth date is February 29
        # and the current year is not a leap year
        birthday = born.replace(year=today.year, day=born.day - 1)
    if birthday > today:
        return today.year - born.year - 1
    else:
        return today.year - born.year


def view_dict(dict_info):
    for k, v in dict_info.items():
        logger.info("%s: %d %s" % (k, len(v), v))
    pass


class GarnetException(Exception):
    def __init__(self, message='Unknown Error！'):
        # self.code = code
        self.msg = message


def save_file(obj, file_path):
    '''
    保存文件
    :param obj: 要保存的对象
    :param file_path: 保存对象的地址
    :return:
    '''
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)
    logger.info("Object saved in: %s", file_path)
    return file_path


def load_file(file_path):
    '''
    从制定的path中加载对象
    :param file_path: 对象的地址
    :return:
    '''
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    logger.info("Loaded file from: %s", file_path)
    return obj


def dump_submission(data, path=config.submission_path):
    logger.info("Shape of result: {}".format(data.shape))
    logger.info("head:\n{}".format(data.head()))
    data.to_csv(path, index=0)
    logger.info("Submission dumped: {}".format(path))


def load_df(path):
    '''
    加载指定路径的数据
    :param path:
    :return: DataFrame
    '''
    fr = codecs.open(path, 'rb', 'utf-8', 'ignore')
    df = pd.read_csv(fr, lineterminator='\n')
    logger.info("Loaded data from: {}".format(path))
    logger.info("Shape of data: {}".format(df.shape))
    if config.DEBUG:
        logger.info("head:\n {}".format(df.head()))
    return df


def encode_one_hot(data, columns):
    '''
    one-hot 编码
    :param data:
    :param columns:
    :return:
    '''
    logger.info("one-hot encode colunmns: {}".format(columns))
    diff = list(set(columns) - set(data.columns))
    features = pd.get_dummies(data=data, dummy_na=True, columns=columns)  # 进行one-hot编码
    if data is not None:
        features = features[:len(data)]
        logger.info('features shape: {}'.format(features.shape))

    # 去除编码后的无效列
    try:
        for name in features.columns:
            if '_nan' in name:
                features.drop([name], axis=1, inplace=True)
                if config.DEBUG:
                    logger.info('drop column: {}'.format(name))
    except Exception as e:
        logger.error(e, exc_info=True)
    return features


def get_age_type(age):
    if age < 30:
        return '1'
    elif age >= 30 and age < 50:
        return '2'
    elif age >= 50 and age < 80:
        return '3'
    else:
        return '4'


def create_counter():
    '''
    https://blog.csdn.net/sinat_41701878/article/details/79301449
    :return: 每次调用自增1
    '''

    def increase():  # 定义一个含有自然数算法的生成器,使用next来完成不断调用的递增
        n = 0
        while True:
            n = n + 1
            yield n

    it = increase()  # 一定要将生成器转给一个(生成器)对象,才可以完成

    def counter():  # 再定义一内函数
        return next(it)  # 调用生成器的值,每次调用均自增

    return counter


if __name__ == "__main__":
    counter = create_counter()
    print(counter())
    print(counter())
