#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-3-12 下午4:40
# @Author  : J.W.
# @File    : feature.py


from sklearn.feature_extraction.text import TfidfVectorizer

import config
from logger import logger
import comm


class TFIDF():
    def __init__(self, data):
        '''
        初始化并拟合数据集
        :param data:
        '''
        self.data = data
        logger.info('init TfidfVectorizer')
        self.tfidf = TfidfVectorizer()
        logger.info('fitting Tfidf...')
        self.train_vec = self.tfidf.fit_transform(data).toarray()
        logger.info('end')

    def transform(self, data):
        '''
        拟合新的数据集
        :param data:
        :return:
        '''
        return self.tfidf.transform(data).toarray()


class Feature():
    def __init__(self):
        self.train_data = comm.load_df(config.train_data_path)
        self.test_data = comm.load_df(config.test_data_path)
        self.tfidf = TFIDF(self.train_data['review'])
        self.y = self.get_target()
        self.X = self.tfidf.train_vec
        logger.info("shape of X: {}".format(self.X.shape))
        logger.info("shape of y: {}".format(self.y.shape))
        self.test_X = self.tfidf.transform(self.test_data['review'])
        self.test_ids = self.test_data["ID"]

    def get_target(self):
        def get_lable(label):
            if label == 'Positive':
                return 1
            else:
                return 0

        self.train_data['y'] = self.train_data.apply(lambda x: get_lable(x['label']), axis=1)
        return self.train_data['y'].values.astype('int')


def tfidf_vec(corpus):
    tfidf = TfidfVectorizer()
    train_vec = tfidf.fit_transform(corpus)
    # for test data
    # tfidf.transform(['ya Allah meri sister Affia ki madad farma', 'khud chahta a is umar main shadi'])
    return train_vec, tfidf

if __name__ == '__main__':
    import comm

    data = comm.load_df('./data/train.csv')
    train_vec, tf = tfidf_vec(data['review'][:200])
    print("\n")
    print(train_vec.toarray()[0])
    print(len(train_vec.toarray()[1]))
    print(len(train_vec.toarray()))

    a = tf.transform(['ya Allah meri sister Affia ki madad farma', 'khud chahta a is umar main shadi'])
    print(a.toarray())
    print(len(a.toarray()))
    print(len(a.toarray()[0]))

    tf = TFIDF(data['review'][:200])
    a = tf.transform(['ya Allah meri sister Affia ki madad farma', 'khud chahta a is umar main shadi'])
    print(a.toarray())
    print(len(a.toarray()))
    print(len(a.toarray()[0]))

    data = comm.load_df(config.train_data_path)

    print(data['y'])
