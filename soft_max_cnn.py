#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-3-13 下午5:27
# @Author  : J.W.
# @File    : test.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-3-13 下午5:27
# @Author  : J.W.
# @File    : test.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from model import SoftMax_CNN, SoftMax_CNN2, TextCNN

import config
from feature import Feature
from  logger import logger

f = Feature()
x = f.X.values
y = f.y
train_word_vec, test_word_vec = f.word_vec()
voc = f.voc

use_cuda = torch.cuda.is_available()
# use_cuda = False
logger.info('use_cuda: {}'.format(use_cuda))



from logger import logger

EMBEDDING_DIM = 6
HIDDEN_DIM = 6
import comm
import numpy as np

corpus_name = 'train'
datafile = './data/train.csv'
train_data = comm.load_df(datafile)


def predict(net, tfidf_vec, word_vec):
    # 预测单条的时候
    result = []
    # nput_s = Variable(torch.FloatTensor(x[k])).unsqueeze(0)
    tfidf_data = Variable(torch.FloatTensor(tfidf_vec)).unsqueeze(0)
    word_data = Variable(torch.LongTensor(word_vec))
    if use_cuda:
        tfidf_data = tfidf_data.cuda()
        word_data = word_data.cuda()
        proba = net(tfidf_data, word_data)
        result.append(list(proba.cpu().data.numpy()[0]))
    else:
        proba = net(tfidf_data, word_data)
        result.append(list(proba.data.numpy()[0]))
    result = np.array(result)
    return result


def train(x, y, word_vec_x):
    epoches = 10
    all_loss = 0
    net = SoftMax_CNN(n_feature=len(x[0]), n_hidden=10, n_out=2, vocb_size=voc.num_words)
    if use_cuda:
        net = net.cuda()
    opitmizer = torch.optim.SGD(net.parameters(), lr=0.03)
    loss_fun = nn.MSELoss()  # 选择 均方差为误差函数
    # loss_fun = nn.NLLLoss()
    for i in range(epoches):
        Length = len(x)
        loss_data = 0
        for k in range(Length):
            input_s = Variable(torch.FloatTensor(x[k])).unsqueeze(0)
            input_word_vec = word_vec_x[k]
            input_word_vec = Variable(torch.LongTensor(input_word_vec))
            if use_cuda:
                input_s = input_s.cuda()
                input_word_vec = input_word_vec.cuda()
            predict = net(input_s, input_word_vec)
            # print(predict)
            target = None
            if (y[k] == 0):
                target = [1, 0]
            else:
                target = [0, 1]
            target = Variable(torch.FloatTensor(target)).unsqueeze(0)  # 变成 1*2的 tensor
            if use_cuda:
                target = target.cuda()
            loss = loss_fun(predict, target)
            loss_data += loss.item()  # 获取tensor的值
            opitmizer.zero_grad()
            loss.backward()
            opitmizer.step()
        avg_loss = loss_data / Length
        logger.info("{} {:4f}".format(i, avg_loss))
        all_loss += avg_loss

    logger.info("{} {:4f}".format(i, all_loss / epoches))
    return net


def cv():
    train_x, test_x, train_y, test_y, train_word_voc, test_word_voc = train_test_split(x, y, train_word_vec.values,
                                                                                       random_state=1,
                                                                                       test_size=0.1)
    logger.info("train/test: {}/{}".format(len(train_y), len(test_y)))
    net = train(train_x, train_y, train_word_voc)
    # net = comm.load_file('./data/model')
    proba = predict_resluts(net,test_x, test_word_voc )
    from report import Report

    r = Report()
    auc = r.report_one_folder(test_y, proba)
    model_path = './data/model-{:.4f}'.format(auc)
    comm.save_file(net, model_path)
    return model_path


def predict_resluts(net, tfidf_vec, word_vec):
    result = []
    for k in range(len(tfidf_vec)):
        proba = predict(net,tfidf_vec[k], word_vec[k])
        result.append(list(proba[0]))

    return np.asarray(result)


def predict_submission(net, path):
    result = predict_resluts(net, f.test_X.values, test_word_vec.values)
    ids = f.test_ids
    assert len(ids) == len(result)
    import pandas as pd
    data = []
    ids = f.test_ids
    for id, p in zip(ids, result):
        data.append([id, p[1]])

    result = pd.DataFrame(data, columns=['ID', "Pred"])
    comm.dump_submission(result, path=path)


def train_all_data():
    '''
    用数据集中所有数据进行模型训练 然后预测结果
    :return:
    '''
    net = train(x, y)
    result_path = './data/result-0.20190314.csv'
    predict_submission(net, result_path)


def train_and_report():
    model_path = cv()
    # model_path = './data/model-0.6145'
    net = comm.load_file(model_path)
    result_path = model_path.replace('model', 'result') + ".csv"
    predict_submission(net, result_path)


if __name__ == "__main__":
    logger.info('use_cuda: {}'.format(use_cuda))
    train_and_report()

    # train_all_data()
