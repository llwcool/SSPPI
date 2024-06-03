import paddle
from TransformerEncoder import TransformerEncoder
from resnet18 import ResNet18
from parameters import weight_attr_1, bias_attr_1

class addAttention(paddle.nn.Layer):
    def __init__(self, dropout = 0.1):
        super(addAttention, self).__init__()
        self.fe_seq = TransformerEncoder(6, 6, 60, 6, 64, dropout, needreshape = False)
        self.fe_dssp = TransformerEncoder(6, 6, 14, 2, 64, dropout, needreshape = False)
        self.fe_cc = TransformerEncoder(6, 6, 28, 4, 64, dropout, needreshape = False)
        self.last_trans = TransformerEncoder(6, 6, 102, 6, 64, dropout, needreshape = False)

    def forward(self, inputs):
        seq = inputs[:, :, :60]
        dssp = inputs[:, :, 60:74]
        cc = inputs[:, :, 74:102]
        seq_feature = self.fe_seq(seq)
        dssp_feature = self.fe_dssp(dssp)
        cc_feature = self.fe_cc(cc)
        out = paddle.concat([seq_feature, dssp_feature, cc_feature], axis=-1) # b × (window_length + struct_window) × 102
        out = self.last_trans(out)
        return seq_feature, dssp_feature, cc_feature, out

class recurrentUnion(paddle.nn.Layer):
    def __init__(self, length, dropout = 0.1):
        super(recurrentUnion, self).__init__()
        self.window_len = 2 * length + 1
        self.conv_shape = 1

        self.addAttention_seqwindow = addAttention(dropout)
        self.addAttention_structwindow = addAttention(dropout)
        self.addAttention_fused = addAttention(dropout)
        self.addAttention_all = addAttention(dropout)
    
    def forward(self, inputs):
        #20, 20, 20, 20, 14, 28
        pssm = inputs[:, :self.window_len, 0:20]
        hmm = inputs[:, :self.window_len, 20:40]
        raw_protein = inputs[:, :self.window_len, 40: 60]
        dssp = inputs[:, :self.window_len, 60:74]
        cc = inputs[:, :self.window_len, 74:102]

        dis_pssm = inputs[:, self.window_len:, 0:20]
        dis_hmm = inputs[:, self.window_len:, 20:40]
        dis_raw_protein = inputs[:, self.window_len:, 40: 60]
        dis_dssp = inputs[:, self.window_len:, 60:74]
        dis_cc = inputs[:, self.window_len:, 74:102]

        input_seqwindow = paddle.concat([pssm, hmm, raw_protein, dssp, cc], axis = -1)
        input_structwindow = paddle.concat([dis_pssm, dis_hmm, dis_raw_protein, dis_dssp, dis_cc], axis = -1)

        seq1, seq2, seq3, out_seqwindow = self.addAttention_seqwindow(input_seqwindow)
        str1, str2, str3, out_structwindow = self.addAttention_structwindow(input_structwindow)
        input_fused = paddle.concat([paddle.concat([seq1, str1], axis = -2), paddle.concat([seq2, str2], axis = -2), paddle.concat([seq3, str3], axis = -2)], axis = -1)
        _, _, _, out_fused = self.addAttention_fused(input_fused)
        out_all = paddle.concat([out_seqwindow, out_structwindow], axis=-2)
        _, _, _, out_all = self.addAttention_all(out_all)
        return out_all, out_fused

class MyNet(paddle.nn.Layer):

    def __init__(self, length, dropout = 0.1):
        super(MyNet, self).__init__()
        self.window_len = 2 * length + 1
        self.conv_shape = 1
        self.unionnum = 1

        self.recurrent_addAttention =  paddle.nn.LayerList([
            recurrentUnion(length, dropout)
            for _ in range(self.unionnum)
        ])

        self.norm = paddle.nn.LayerList([paddle.nn.LayerNorm(102) for _ in range(self.unionnum)])

        self.res = ResNet18(1,256)

        self.relu = paddle.nn.ReLU()
        self.fc11 = paddle.nn.Linear(256, 128, weight_attr=weight_attr_1, bias_attr=bias_attr_1)
        self.fc12 = paddle.nn.Linear(128, 32, weight_attr=weight_attr_1, bias_attr=bias_attr_1)
        self.fc13 = paddle.nn.Linear(32, 1, weight_attr=weight_attr_1, bias_attr=bias_attr_1)
        
    def forward(self, inputs):
        #20, 20, 20, 20, 14, 28
        pssm = inputs[:, :self.window_len, 0:20] + inputs[:, :self.window_len, 60: 80]
        hmm = inputs[:, :self.window_len, 20:40] + inputs[:, :self.window_len, 60: 80]
        raw_protein = inputs[:, :self.window_len, 40: 60] + inputs[:, :self.window_len, 60: 80]
        dssp = inputs[:, :self.window_len, 80:94]
        cc = inputs[:, :self.window_len, 94:122]

        dis_pssm = inputs[:, self.window_len:, 0:20]
        dis_hmm = inputs[:, self.window_len:, 20:40]
        dis_raw_protein = inputs[:, self.window_len:, 40: 60]
        dis_dssp = inputs[:, self.window_len:, 80:94]
        dis_cc = inputs[:, self.window_len:, 94:122]

        input_seqwindow = paddle.concat([pssm, hmm, raw_protein, dssp, cc], axis = -1)
        input_structwindow = paddle.concat([dis_pssm, dis_hmm, dis_raw_protein, dis_dssp, dis_cc], axis = -1)

        init_input = paddle.concat([input_seqwindow, input_structwindow], axis = -2)
        out = init_input
        i = 0
        for layer, norm in zip(self.recurrent_addAttention, self.norm):
            out_all, out_fused = layer(out)
            i += 1
            if i != self.unionnum:
                out = out_fused + out_all
                out = norm(out)
            else:
                out_all = norm(out_all)
                out_fused = norm(out_fused)
        b, c, n = out_all.shape
        out_all = paddle.reshape(out_all, shape = [b, 1, c, n])
        out_fused = paddle.reshape(out_fused, shape = [b, 1, c, n])
        # out = paddle.concat([out_all, out_fused], axis = 1)
        out = out_fused
        out = self.res(out)
        out = self.relu(self.fc11(out))
        out = self.relu(self.fc12(out))
        out = self.fc13(out)
        return out
