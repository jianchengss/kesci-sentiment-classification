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


class SoftMax(nn.Module):
    def __init__(self, n_feature, n_hidden, n_out):
        super(SoftMax, self).__init__()
        self.hidden = nn.Linear(n_feature, n_hidden)
        self.out = nn.Linear(n_hidden, n_out)

    def forward(self, x):
        x = torch.sigmoid(self.hidden(x))
        x = self.out(x)
        return F.softmax(x, dim=1)  # 返回的是每个类的概率


from logger import logger

EMBEDDING_DIM = 6
HIDDEN_DIM = 6
import comm
from word_vec import loadPrepareData

corpus_name = 'train'
datafile = './data/train.csv'
train_data = comm.load_df(datafile)
voc = loadPrepareData(train_data, corpus_name)
from feature import Feature


def predict(x, net):
    # 预测单条的时候
    # for data in x:
    #     proba = net(Variable(torch.FloatTensor(data)).unsqueeze(0))
    proba = net(Variable(torch.FloatTensor(x)))
    return proba.data.numpy()


def train(x, y):
    epoches = 100
    all_loss = 0
    net = SoftMax(n_feature=len(x[0]), n_hidden=10, n_out=2)
    opitmizer = torch.optim.SGD(net.parameters(), lr=0.03)
    loss_fun = nn.MSELoss()  # 选择 均方差为误差函数
    for i in range(epoches):
        Length = len(x)
        loss_data = 0
        for k in range(Length):
            input_s = Variable(torch.FloatTensor(x[k])).unsqueeze(0)
            predict = net(input_s)
            # print(predict)
            target = None
            if (y[k] == 0):
                target = [1, 0]
            else:
                target = [0, 1]
            target = Variable(torch.FloatTensor(target)).unsqueeze(0)  # 变成 1*2的 tensor
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


f = Feature()
x = f.X.values
y = f.y


def cv():
    train_x, test_x, train_y, test_y = train_test_split(x, y, random_state=1, test_size=0.2)
    net = train(train_x, train_y)
    # net = comm.load_file('./data/model')
    proba = predict(test_x, net)
    from report import Report

    r = Report()
    auc = r.report_one_folder(test_y, proba)
    model_path = './data/model-{:4f}'.format(auc)
    comm.save_file(net, model_path)
    return  model_path


def predict_submission(net, path):
    import pandas as pd
    data = []
    ids = f.test_ids
    proba = predict(f.test_X.values, net)
    assert len(ids) == len(proba)
    for id, p in zip(ids, proba):
        data.append([id, p[1]])

    result = pd.DataFrame(data, columns=['ID', "Pred"])
    comm.dump_submission(result, path=path)


if __name__ == "__main__":
    model_path = cv()

    # model_path = './data/model-0.8369'
    net = comm.load_file(model_path)
    result_path = model_path.replace('model', 'result') + ".csv"
    predict_submission(net, result_path)
