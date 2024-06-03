import paddle
from paddle.nn import Conv2D, BatchNorm
import paddle.nn.functional as F
from parameters import weight_attr_1, bias_attr_1


class ResNetBlock(paddle.nn.Layer):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()

        self.conv1 = Conv2D(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, weight_attr = weight_attr_1, bias_attr=False)
        self.bn1 = BatchNorm(out_channels)
        self.conv2 = Conv2D(out_channels, out_channels, kernel_size=3, stride=1, padding=1, weight_attr = weight_attr_1, bias_attr=False)
        self.bn2 = BatchNorm(out_channels)

        self.shortcut = paddle.nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = paddle.nn.Sequential(
                Conv2D(in_channels, out_channels, kernel_size=1, stride=stride, weight_attr = weight_attr_1, bias_attr=False),
                BatchNorm(out_channels)
            )

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        y += self.shortcut(x)
        return F.relu(y)

class ResNet18(paddle.nn.Layer):
    def __init__(self, input_dim, output_dim):
        super(ResNet18, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.conv1 = Conv2D(self.input_dim, 8, kernel_size=3, stride=1, padding=1, bias_attr=False)
        self.bn1 = BatchNorm(8)
        self.layer1 = paddle.nn.Sequential(
            ResNetBlock(8, 8),
            ResNetBlock(8, 8)
        )
        self.layer2 = paddle.nn.Sequential(
            ResNetBlock(8, 16, stride=2),
            ResNetBlock(16, 16)
        )
        self.layer3 = paddle.nn.Sequential(
            ResNetBlock(16, 32, stride=2),
            ResNetBlock(32, 32)
        )
        self.layer4 = paddle.nn.Sequential(
            ResNetBlock(32, 64, stride=2),
            ResNetBlock(64, 64)
        )
        self.layer5 = paddle.nn.Sequential(
            ResNetBlock(64, 128, stride=2),
            ResNetBlock(128, 128)
        )
        self.layer6 = paddle.nn.Sequential(
            ResNetBlock(128, 256, stride=2),
            ResNetBlock(256, 256)
        )
        self.avg_pool = paddle.nn.AdaptiveAvgPool2D((1, 1))
        self.to_out = paddle.nn.Linear(256, self.output_dim, weight_attr = weight_attr_1, bias_attr=bias_attr_1)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = F.relu(y)
        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)
        y = self.layer5(y)
        y = self.layer6(y)
        y = self.avg_pool(y)
        y = paddle.flatten(y, 1)
        y = self.to_out(y)
        return y
