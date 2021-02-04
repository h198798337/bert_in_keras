#!/usr/bin/env python3
# Author: fanqiang
# create date: 2021/1/28
# Content:
# desc:
# !/usr/bin/env python3
# Author: fanqiang
# create date: 2021/1/27
# Content: 主题判断
# desc:
import re, os
os.environ.setdefault('TF_KERAS', '1')

from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import codecs
import tensorflow as tf
import time
import numpy as np
import pandas as pd
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab

g_base_path = os.path.dirname(os.path.realpath(__file__))

maxlen = 500

test_data = []
for line in open(g_base_path + '/resource/功能维度相对人工判错样本.csv', "r", encoding="utf-8"):
    if line.strip() != '':
        test_data.append(line.split('|'))
test_data = np.array(test_data)


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


config_path = '/data/home/fanqiang/dm/script/resources/nezha-base/bert_config.json'
checkpoint_path = '/data/home/fanqiang/dm/script/resources/nezha-base/model.ckpt-900000'
dict_path = '/data/home/fanqiang/dm/script/resources/nezha-base/vocab.txt'
token_dict = {}

# 令牌器改造
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
                R.append('[unused1]')  # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]')  # 剩余的字符是[UNK]
        return R


tokenizer = OurTokenizer(token_dict)

# 构造模型
bert_model = build_transformer_model(config_path, checkpoint_path, model='nezha')

for l in bert_model.layers:
    l.trainable = True

t1_in = Input(shape=(None,))
t2_in = Input(shape=(None,))

t = bert_model([t1_in, t2_in])
t = Lambda(lambda x: x[:, 0])(t)  # 这里为什么这么暴力，直接就是取第一个，这是bert模型的性质决定的，bert模型输出的第一个值，是cls这个标签的相似值，如果是做推断任务，用这个cls标签的值就足够了
ps1 = Dense(1, activation='sigmoid')(t)

subject_model = Model([t1_in, t2_in], [ps1])  # 预测subject的模型
subject_model.summary()

subject_model.load_weights('/data/home/fanqiang/bert_in_keras/model/subject_model_nezha/subject_model_nezha')


def pred_result_convert(key_sentence, sentence):
    # 根据标点符号分割
    pattern = '？|\s|,|\.|/|;|\'|`|\[|\]|<|>|\?|:|"|\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|-|=|\_|\+|，|。|、|；|‘|’|【|】|·|！| |…|（|）'
    split_sentence = re.split(pattern, sentence[1].strip())
    X1, X2 = [], []
    for s in split_sentence:
        if s.strip() == '':
            continue
        x1, x2 = tokenizer.encode(key_sentence, s)
        X1.append(x1)
        X2.append(x2)
    # x1, x2 = tokenizer.encode(key_sentence, sentence[1].strip())
    # X1.append(x1)
    # X2.append(x2)
    X1 = seq_padding(X1)
    X2 = seq_padding(X2)
    y_pred = subject_model.predict([X1, X2])
    for y in y_pred:
        if y[0] > 0.7:
            return 1, y_pred
    return 0, y_pred


test_result = []
for d in test_data:
    test_result.append(pred_result_convert('使用效果', d))
test_result = np.array(test_result)

# 找出错误分类的索引
y_pred = [int(d) for d in test_data[:, 0]]
misclassified = np.where(test_result[:, 0] != y_pred)
# 输出所有错误分类的索引
print("测试集:样本总数{},估错样本数{},错率{}%".format(len(test_data), len(misclassified[0]), len(misclassified[0])  * 100.0 / len(test_data)))

for i in misclassified[0]:
    print('------------------------------------------')
    print(test_data[i][0], test_data[i][1].strip())
    print(test_result[i][0], ','.join([str(j) for j in test_result[i][1]]))