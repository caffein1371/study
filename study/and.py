# -*- coding: UTF-8 -*-

import random
import argparse
import numpy
import chainer
import chainer.optimizers
import matplotlib.pyplot as plt


class SMallClassificationModel(chainer.FunctionSet):
    #入力2ベクトル，出力2ベクトル
    #ニューロン群
    def __init__(self):
        super(SMallClassificationModel,self).__init__(
            fc1 = chainer.functions.Linear(2,2)
            )
    #どう結合するか
    def _forward(self,x):
        h = self.fc1(x)
        return h

    #訓練
    def train(self, x_data, y_data):
        #reshape(1,2)より　１つの１次元配列を1つの２次元配列にする
        x = chainer.Variable(x_data.reshape(1,2).astype(numpy.float32),volatile=False)
        #yは正解データ
        y = chainer.Variable(y_data.astype(numpy.int32),volatile=False)
        h = self._forward(x)

        #勾配法の初期化
        optimizer.zero_grads()
        #コスト関数をソフトマックス関数を用いて演算
        error = chainer.functions.softmax_cross_entropy(h,y)
        error.backward()
        optimizer.update()

        print("x: {}".format(x.data))
        print("y: {}".format(y.data))
        print("h: {}".format(h.data))
        print("h_class: {}".format(h.data.argmax()))

model = SMallClassificationModel()
optimizer = chainer.optimizers.MomentumSGD(lr=0.01, momentum=0.9)
optimizer.setup(model.collect_parameters())

#4パターン×1000個のデータ
data_and = [
    [numpy.array([0,0]), numpy.array([0])],
    [numpy.array([0,1]), numpy.array([0])],
    [numpy.array([1,0]), numpy.array([0])],
    [numpy.array([1,1]), numpy.array([1])],
]*1000

for invec, outvec in data_and:
    model.train(invec, outvec)
