import math
import numpy as np
import paddle
import paddle.nn as nn
from paddle.regularizer import L1Decay, L2Decay
from paddle import ParamAttr
from TransformerEncoder import TransformerEncoder, FeedForward
from resnet18 import ResNet18

weight_attr_1 = ParamAttr(initializer = nn.initializer.XavierNormal(), regularizer = L2Decay(0.005))
bias_attr_1 = ParamAttr(initializer = nn.initializer.XavierNormal(), regularizer = L2Decay(0.005))


class MyNet(paddle.nn.Layer):

    def __init__(self, length, dropout = 0.1):
        super(MyNet, self).__init__()
        self.length = length
        self.window_len = 2 * self.length + 1
        self.conv_shape = 1

        self.fe_seq = nn.Sequential(
            TransformerEncoder(6, 6, 60, 6, 50, dropout),
            ResNet18(1, 60),
            paddle.nn.Dropout(dropout)
        )
        self.fe_dssp = nn.Sequential(
            TransformerEncoder(6, 6, 14, 2, 50, dropout),
            ResNet18(1, 14),
            paddle.nn.Dropout(dropout)
        )
        self.fe_cc = nn.Sequential(
            TransformerEncoder(6, 6, 28, 4, 50, dropout),
            ResNet18(1, 28),
            paddle.nn.Dropout(dropout)
        )
        self.relu1 = paddle.nn.ReLU()
        self.relu2 = paddle.nn.ReLU()
        self.fc11 = paddle.nn.Linear(102, 64, weight_attr=weight_attr_1, bias_attr=bias_attr_1)
        self.fc12 = paddle.nn.Linear(64, 32, weight_attr=weight_attr_1, bias_attr=bias_attr_1)
        self.fc13 = paddle.nn.Linear(32, 1, weight_attr=weight_attr_1, bias_attr=bias_attr_1)
        
    def forward(self, inputs):
        #20, 20, 20, 20, 14, 28
        pssm = inputs[:, :, 0:20] + inputs[:, :, 60: 80]
        hmm = inputs[:, :, 20:40] + inputs[:, :, 60: 80]
        raw_protein = inputs[:, :, 40: 60] + inputs[:, :, 60: 80]
        dssp = inputs[:, :, 80:94]
        cc = inputs[:, :, 94:122]
        
        seq = paddle.concat([pssm, hmm, raw_protein], axis = -1)
        # 11, 20
        seq_feature = self.fe_seq(seq)
        dssp_feature = self.fe_dssp(dssp)
        cc_feature = self.fe_cc(cc)

        out = paddle.concat([seq_feature, dssp_feature, cc_feature], axis=-1)
        out = self.relu1(self.fc11(out))
        out = self.relu2(self.fc12(out))
        out = self.fc13(out)
        return out
