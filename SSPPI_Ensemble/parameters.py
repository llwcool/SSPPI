import paddle
from paddle import ParamAttr
import paddle.nn as nn
from paddle.regularizer import L2Decay

def getLoss(predicts, y_data, fg_num, alpha = 0.85):
    return paddle.nn.functional.sigmoid_focal_loss(
                predicts, y_data, normalizer=fg_num, alpha=alpha)

weight_attr_1 = ParamAttr(initializer = nn.initializer.XavierNormal(), regularizer = L2Decay(0.005))
bias_attr_1 = ParamAttr(initializer = nn.initializer.XavierNormal(), regularizer = L2Decay(0.005))
