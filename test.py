#!/usr/bin/env python3
# Author: fanqiang
# create date: 2021/1/26
# Content: 
# desc:

from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras.layers import Dense, Input, Lambda
from keras.models import Sequential, Model
import json
import numpy as np
from random import choice
from tqdm import tqdm
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import re, os
import codecs
import tensorflow as tf

# t1_in = Input(shape=(1,))
# t2_in = Input(shape=(1,))
#
# ps1 = Dense(1, activation='sigmoid')(t1_in)
# # ps2 = Dense(1, activation='sigmoid')(t2_in)
#
# model = Model([t1_in], [ps1])
# model.summary()
#
# test_data = [[1]]
# pr1 = model.predict([test_data])
# print(pr1)


config_path = '/data/home/fanqiang/dm/script/resources/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/data/home/fanqiang/dm/script/resources/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/data/home/fanqiang/dm/script/resources/chinese_L-12_H-768_A-12/vocab.txt'
token_dict = {}

with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]') # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]') # 剩余的字符是[UNK]
        return R

tokenizer = OurTokenizer(token_dict)

bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)

for l in bert_model.layers:
    l.trainable = True

t1_in = Input(shape=(None,))
t2_in = Input(shape=(None,))

t = bert_model([t1_in, t2_in])
# t = Lambda(lambda x: x[:, 0])(t) # 这里为什么这么暴力，直接就是取第一个，这是bert模型的性质决定的，bert模型输出的第一个值，是cls这个标签的相似值，如果是做推断任务，用这个cls标签的值就足够了
ps1 = Dense(1, activation='sigmoid')(t)

subject_model = Model([t1_in, t2_in], [ps1]) # 预测subject的模型
subject_model.summary()

text_in = '《冬天的故事》是张睿的音乐作品，很好听'
text_in_2 = '不错'

_tokens = tokenizer.tokenize(text_in, text_in_2)
print(_tokens)
_t1, _t2 = tokenizer.encode(first=text_in, second=text_in_2)
_t1, _t2 = np.array([_t1]), np.array([_t2])
# _t1, _t2 = np.expand_dims(_t1, axis=2), np.expand_dims(_t2, axis=2)
_k1 = subject_model.predict([_t1, _t2])
print(_k1)
