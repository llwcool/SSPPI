import paddle
from parameters import weight_attr_1, bias_attr_1


class TransformerEncoder(paddle.nn.Layer):
    def __init__(self, encoder_layers, nlayers, dim, heads, dim_ff, dropout=0.1, needreshape = True):
        super().__init__()
        self.dim = dim
        self.needreshape = needreshape
        self.layers = paddle.nn.LayerList([
            TransformerEncoderLayer(dim=dim, heads=heads, dim_ff=dim_ff, dropout=dropout)
            for _ in range(encoder_layers)
        ])

        self.norm = paddle.nn.LayerList([paddle.nn.LayerNorm(dim) for _ in range(nlayers)])

    def forward(self, x):
        b, c, n = x.shape
        for layer, norm in zip(self.layers, self.norm):
            x = layer(x)
            x = norm(x)
        if self.needreshape:
            x = paddle.reshape(x, shape = [b, 1, c, n])
        return x

class TransformerEncoderLayer(paddle.nn.Layer):
    def __init__(self, dim, heads, dim_ff, dropout):
        super().__init__()
        self.dim = dim
        self.self_attn = SelfAttention(dim=dim, heads=heads)
        self.feed_forward = FeedForward(dim=dim, dim_ff=dim_ff, dropout=dropout)

        self.norm1 = paddle.nn.LayerNorm(dim)
        self.norm2 = paddle.nn.LayerNorm(dim)

        self.dropout1 = paddle.nn.Dropout(dropout)
        self.dropout2 = paddle.nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout1(self.self_attn(x))
        x = self.norm1(x)

        x = x + self.dropout2(self.feed_forward(x))
        x = self.norm2(x)
        return x

class SelfAttention(paddle.nn.Layer):
    def __init__(self, dim, heads):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.scale = dim ** -0.5
        self.to_qkv = paddle.nn.Linear(dim, dim * 3, weight_attr=weight_attr_1, bias_attr= False)
        self.to_out = paddle.nn.Linear(dim, dim, weight_attr=weight_attr_1, bias_attr= bias_attr_1)
        
    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, axis=-1)
        q, k, v = [paddle.transpose(paddle.reshape(t, [b, -1, self.heads, self.dim // self.heads]), [0, 2, 1, 3]) for t in qkv]
        attn = paddle.matmul(q, k.transpose([0, 1, 3, 2])) * self.scale
        attn = paddle.nn.functional.softmax(attn, axis=-1)
        z = paddle.matmul(attn, v)
        z = paddle.reshape(z, [b, c, n])
        z = self.to_out(z)
        return z

class FeedForward(paddle.nn.Layer):
    def __init__(self, dim, dim_ff, dropout):
        super().__init__()

        self.net = paddle.nn.Sequential(
            paddle.nn.Linear(dim, dim_ff, weight_attr=weight_attr_1, bias_attr= bias_attr_1),
            paddle.nn.ReLU(),
            paddle.nn.Dropout(dropout),
            paddle.nn.Linear(dim_ff, dim, weight_attr=weight_attr_1, bias_attr= bias_attr_1),
            paddle.nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
