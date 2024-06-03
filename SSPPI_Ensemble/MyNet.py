import paddle
from parameters import weight_attr_1, bias_attr_1

class MyNet(paddle.nn.Layer):

    def __init__(self, dropout = 0.1):
        super(MyNet, self).__init__()
        self.relu = paddle.nn.ReLU()

        self.fc11 = paddle.nn.Linear(3 * 33 + 122, 128, weight_attr=weight_attr_1, bias_attr=bias_attr_1)
        self.fc12 = paddle.nn.Linear(128, 32, weight_attr=weight_attr_1, bias_attr=bias_attr_1)
        self.fc13 = paddle.nn.Linear(32, 1, weight_attr=weight_attr_1, bias_attr=bias_attr_1)
        self.dropout1 = paddle.nn.Dropout(dropout)
    def forward(self, inputs):
        local_all = inputs[:, :32]
        local_onlyseq = inputs[:, 32: 64]
        local_onlystr = inputs[:, 64: 96]
        protein_feature = inputs[:, 96: 218]
        protein_length = inputs[:, 218]
        protein_length = paddle.unsqueeze(protein_length, axis = 1)

        local_all = paddle.concat([local_all, protein_length], axis = -1)
        local_onlyseq = paddle.concat([local_onlyseq, protein_length], axis = -1)
        local_onlystr = paddle.concat([local_onlystr, protein_length], axis = -1)

        local_feature = paddle.concat([local_all, local_onlyseq, local_onlystr, protein_feature], axis = 1)
        
        out = self.dropout1(self.relu(self.fc11(local_feature)))
        out = self.dropout1(self.relu(self.fc12(out)))
        out = self.fc13(out)
        return out
