# -*- coding: utf-8 -*-

"""
Chatbot Tutorial
================
**Author:** `Matthew Inkawhich <https://github.com/MatthewInkawhich>`_

http://fancyerii.github.io/2019/02/14/chatbot/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import itertools
import unicodedata

import os
import pandas as pd
import re
import torch

import comm
import config
from logger import logger

USE_CUDA = torch.cuda.is_available()
USE_CUDA = False
device = torch.device("cuda" if USE_CUDA else "cpu")

corpus = 'data'


def printLines(file, n=10):
    df_result = comm.load_df(file)
    logger.info("\n : {}".format(df_result.head(n)))


printLines(os.path.join(corpus, "train.csv"))

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token


class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # Count SOS, EOS, PAD

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        logger.info('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # Count default tokens

        for word in keep_words:
            self.addWord(word)


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


# Returns True iff both sentences in a pair 'p' are under the MAX_LENGTH threshold
def filter(p):
    # Input sequences need to preserve the last word for EOS token
    return len(p[0].split(' ')) < config.word_max_length and len(p[1].split(' ')) < config.word_max_length


# Using the functions defined above, return a populated voc object and pairs list
def loadPrepareData(data, corpus_name):
    logger.info("Start preparing training data ...")
    voc = Voc(corpus_name)
    for line in data['review']:
        if filter(line):
            voc.addSentence(line)
    logger.info("Counted words: {}".format(voc.num_words))
    return voc


corpus_name = 'train'
datafile = './data/train.csv'
train_data = comm.load_df(datafile)
voc = loadPrepareData(train_data, corpus_name)

MIN_COUNT = 3  # Minimum word count threshold for trimming


def trimRareWords(voc, pairs, MIN_COUNT):
    # Trim words used under the MIN_COUNT from the voc
    voc.trim(MIN_COUNT)
    # Filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # Check input sentence
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        # Check output sentence
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)

    logger.info("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs),
                                                                      len(keep_pairs) / len(pairs)))
    return keep_pairs


def indexesFromSentence(voc, sentence):
    vec = [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]
    return vec


# for sentence in train_data['review']:
#     idx = indexesFromSentence(voc, sentence)
# logger.info('end')


def zeroPadding(data, fillvalue=PAD_token):
    padded_data = itertools.zip_longest(*data, fillvalue=fillvalue)
    return list(padded_data)


def padding(data, voc, fillvalue=PAD_token):
    vec = [indexesFromSentence(voc, sentence) for sentence in data]
    result = []
    for v in vec:
        if len(v) < config.word_max_length:
            while len(v) < config.word_max_length:
                v.append(PAD_token)
            result.append(v)
        elif len(v) > config.word_max_length:
            result.append(v[:config.word_max_length])
        else:
            result.append(v)

    return result


def inputVec(data, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in data]
    padList = zeroPadding(indexes_batch)
    return padList


def get_word_vec(data):
    vec = padding(train_data['review'], voc)
    vec_df = pd.DataFrame(vec)
    logger.info("shape of vec: {}".format(vec_df.shape))
    return vec_df


if __name__ == "__main__":
    input_vec = padding(train_data['review'], voc)
    logger.info("{} {}".format(len(input_vec[1]), input_vec[1]))
    logger.info("{} {}".format(len(input_vec[2]), input_vec[2]))

    input_df = pd.DataFrame(input_vec)
    logger.info(input_df.shape)

    get_word_vec(train_data['review'])
