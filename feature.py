#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-3-12 下午4:40
# @Author  : J.W.
# @File    : feature.py


import comm
import config
import pandas as pd
import word_vec
from logger import logger
from sklearn.feature_extraction.text import TfidfVectorizer


class TFIDF():
    def __init__(self, data, min_df=1):
        '''
        初始化并拟合数据集
        :param data:
        '''
        self.data = data
        logger.info('init TfidfVectorizer')
        self.tfidf = TfidfVectorizer(min_df=min_df)
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
        self.test_ids = self.test_data["ID"]
        self.y = self.get_target()

        self.train_features = []
        self.test_features = []

        self.tfidf_vec()
        self.word_vec()  # 加了会很低

        self.X = pd.concat(self.train_features, axis=1)
        self.test_X = pd.concat(self.test_features, axis=1)
        logger.info("Shape of train X: {}".format(self.X.shape))
        logger.info("Shape of test X: {}".format(self.test_X.shape))
        logger.info("Shape of y: {}".format(self.y.shape))

    def get_target(self):
        def get_lable(label):
            if label == 'Positive':
                return 1
            else:
                return 0

        self.train_data['y'] = self.train_data.apply(lambda x: get_lable(x['label']), axis=1)
        return self.train_data['y'].values.astype('int')

    def tfidf_vec(self):
        logger.info("start collect tfidf vec.")
        tfidf = TFIDF(self.train_data['review'], min_df=config.tfidf_min_df)
        train_vec = tfidf.train_vec
        test_vec = tfidf.transform(self.test_data['review'])
        logger.info("shape of trian tfidf: {}".format(train_vec.shape))
        logger.info("shape of test tfidf: {}".format(test_vec.shape))

        self.train_features.append(pd.DataFrame(train_vec))
        self.test_features.append(pd.DataFrame(test_vec))

    def word_vec(self):
        logger.info("word vec")
        train_word_vec = word_vec.get_word_vec(self.train_data['review'])
        test_word_vec = word_vec.get_word_vec(self.test_data['review'])

        logger.info("shape of trian word_vec: {}".format(train_word_vec.shape))
        logger.info("shape of test word_vec: {}".format(test_word_vec.shape))

        self.train_features.append(pd.DataFrame(train_word_vec))
        self.test_features.append(pd.DataFrame(test_word_vec))


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
