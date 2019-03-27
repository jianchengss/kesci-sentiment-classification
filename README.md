# kesci-sentiment-classification

[文本情感分类模型搭建 | 练习赛](https://www.kesci.com/home/competition/5c77ab9c1ce0af002b55af86)


## 实验过程解析

https://www.jiancheng.ai/2019/03/15/kesci-sentiment-classification/

## 赛题描述

本练习赛所用数据，是名为「Roman Urdu DataSet」的公开数据集。

这些数据，均为文本数据。原始数据的文本，对应三类情感标签：Positive, Negative, Netural。

本练习赛，移除了标签为Netural的数据样例。因此，练习赛中，所有数据样例的标签为Positive和Negative。

本练习赛的任务是「分类」。「分类目标」是用训练好的模型，对测试集中的文本情感进行预测，判断其情感为「Negative」或者「Positive」。

## 提交结果

实验结果摘要：

Date    | SHA     | Method |AUC-train  |AUC-kesci| P | R |F
---     |---      |---     |---|---|---|---|---
20190307|8225844  |random   |-      |0.5057 | - | -| -|
20190312|05a6790  |rfc-1    |0.8130 |0.8054 |0.7516 |0.7542 | 0.7520
20190312|d4e00a1  |neural_clf|0.8336 |0.8357 |0.7831 |0.7638 | 0.7728
20190313|9dae25e  |forest-2 |0.8269 |0.8229 |0.7602 |0.7726 | 0.7656
20190313|78f2dc7  |neural_clf |0.8352 |0.8412 |0.7723 |0.7806 | 0.7762
20190313|14075a4  |soft_max |0.8369 |0.8250 |0.7574 |0.7911 | 0.7739
20190313|a592ca6  |soft_max |__0.8641__ |0.8568 |0.7930 |0.8000 | 0.7965
20190313|257fbab  |soft_max |0.8520 |0.8474 |0.7368 |0.8252 | 0.7785
20190313|25b5456  |soft_max |-      |__0.8658__ |- |- | -
