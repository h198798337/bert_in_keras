#!/usr/bin/env python3
# Author: fanqiang
# create date: 2021/2/3
# Content: 
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
from common.dingding_warn import DingDingWarn

ddw = DingDingWarn('951e7381a4f2c3bd1b307ca72e47ed8e2582b76bc1ff36e541fcfb8c120884db',
                   'SEC9deb3e4206872ae86df0e3d0f30bfdb5039421e39fb69233a7deb9a8a21f1d67')

g_base_path = os.path.dirname(os.path.realpath(__file__))

maxlen = 500

train_data = []
for line in open(g_base_path + '/resource/train_data.txt', "r", encoding="utf-8"):
    if line.strip() != '':
        train_data.append(line.split(' '))

valid_data = []
for line in open(g_base_path + '/resource/valid_data.txt', "r", encoding="utf-8"):
    if line.strip() != '':
        valid_data.append(line.split(' '))


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


class data_generator:
    def __init__(self, data, batch_size=32):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i]
                text_1, text_2 = d[0][:maxlen], d[1][:maxlen]
                x1, x2 = tokenizer.encode(text_1, text_2)
                y = float(d[2])
                X1.append(x1)
                X2.append(x2)
                Y.append([y])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    yield [X1, X2], Y
                    [X1, X2, Y] = [], [], []


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
subject_model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(1e-5),  # 用足够小的学习率
    metrics=['accuracy']
)
subject_model.summary()

# 定义early stoping如果3个epoch内validation loss没有改善则停止训练
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
# 自动降低learning rate
lr_reduction = ReduceLROnPlateau(monitor='val_loss',
                                 factor=0.1, min_lr=1e-6, patience=0,
                                 verbose=1)
# 定义callback函数
callbacks = [
    early_stopping,
    lr_reduction
]

start_time = time.time()
# 开始训练
train_D = data_generator(train_data)
valid_D = data_generator(valid_data)

subject_model.fit_generator(
    train_D.__iter__(),
    steps_per_epoch=len(train_D),
    epochs=5,
    validation_data=valid_D.__iter__(),
    validation_steps=len(valid_D)
)
# evaluate_result = subject_model.evaluate(x_test, y_test)
# print('Accuracy:{0:.2%}'.format(evaluate_result[1]))

# 训练好的模型保存到临时文件
subject_model.save_weights('/data/home/fanqiang/bert_in_keras/model/subject_model_nezha/subject_model_nezha')
cost_time = time.time() - start_time
print('cost_time:{}'.format(cost_time))
ddw.send_warning_msg('主题模型训练任务已完成', '耗时：' + str(cost_time))
