import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F


def model_A(num_classes):
    model_resnet = models.resnet18(pretrained=False)
    num_features = model_resnet.fc.in_features
    model_resnet.fc = nn.Linear(num_features, num_classes)
    return model_resnet


# Residual模块定义
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.gelu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.gelu(Y)


# 残差块
def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


# Inception模块定义
class Inception(nn.Module):
    # c1--c4是每条路径的输出通道数
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 线路1，单1x1卷积层
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 线路2，1x1卷积层后接3x3卷积层
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3，1x1卷积层后接5x5卷积层
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4，3x3最大汇聚层后接1x1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.gelu(self.p1_1(x))
        p2 = F.gelu(self.p2_2(F.gelu(self.p2_1(x))))
        p3 = F.gelu(self.p3_2(F.gelu(self.p3_1(x))))
        p4 = F.gelu(self.p4_2(self.p4_1(x)))
        # 在通道维度上连结输出
        return torch.cat((p1, p2, p3, p4), dim=1)


# Dense块定义
class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(
                num_channels * i + input_channels, num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # 连接通道维度上每个块的输入和输出
            X = torch.cat((X, Y), dim=1)
        return X


# 卷积块辅助函数
def conv_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.GELU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1))


# 过渡层，使用1x1卷积降低通道数
def transition_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.GELU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2))


def model_B(num_classes):
    r1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                       nn.BatchNorm2d(64), nn.GELU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    r2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
    r3 = nn.Sequential(*resnet_block(64, 128, 2))
    r4 = nn.Sequential(*resnet_block(128, 256, 2))
    r5 = nn.Sequential(*resnet_block(256, 512, 2))
    netr = nn.Sequential(r1, r2, r3, r4, r5,
                         nn.AdaptiveAvgPool2d((1, 1)),
                         nn.Flatten(), nn.Linear(512, num_classes))

    g1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                       nn.GELU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    g2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                       nn.GELU(),
                       nn.Conv2d(64, 192, kernel_size=3, padding=1),
                       nn.GELU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    g3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                       Inception(256, 128, (128, 192), (32, 96), 64),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    g4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                       Inception(512, 160, (112, 224), (24, 64), 64),
                       Inception(512, 128, (128, 256), (24, 64), 64),
                       Inception(512, 112, (144, 288), (32, 64), 64),
                       Inception(528, 256, (160, 320), (32, 128), 128),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    g5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                       Inception(832, 384, (192, 384), (48, 128), 128),
                       nn.AdaptiveAvgPool2d((1, 1)),
                       nn.Flatten())

    netg = nn.Sequential(g1, g2, g3, g4, g5, nn.Linear(1024, 10))

    d1 = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64), nn.GELU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    # num_channels为当前的通道数
    num_channels, growth_rate = 64, 32
    num_convs_in_dense_blocks = [4, 4, 4, 4]
    blks = []
    for i, num_convs in enumerate(num_convs_in_dense_blocks):
        blks.append(DenseBlock(num_convs, num_channels, growth_rate))
        # 上一个稠密块的输出通道数
        num_channels += num_convs * growth_rate
        # 在稠密块之间添加一个转换层，使通道数量减半
        if i != len(num_convs_in_dense_blocks) - 1:
            blks.append(transition_block(num_channels, num_channels // 2))
            num_channels = num_channels // 2

    netd = nn.Sequential(
        d1, *blks,
        nn.BatchNorm2d(num_channels), nn.GELU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(num_channels, 10))

    # 组合Residual模块、Inception模块、Dense模块定义网络，激活函数采用GELU
    layer1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),  # 尺寸 112
                           nn.BatchNorm2d(64), nn.GELU(),
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))  # 尺寸 56

    layer2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                           nn.GELU(),
                           nn.Conv2d(64, 192, kernel_size=3, padding=1),
                           nn.GELU(),
                           nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
    # 经过layer3，通道数为480
    layer3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                           Inception(256, 128, (128, 192), (32, 96), 64),
                           nn.MaxPool2d(kernel_size=3, stride=1, padding=1))  # 尺寸 56

    # 引入Dense模块
    num_channels, growth_rate = 480, 40
    num_convs_in_dense_blocks = [4, 4, 4]
    blks = []
    for i, num_convs in enumerate(num_convs_in_dense_blocks):
        blks.append(DenseBlock(num_convs, num_channels, growth_rate))
        # 上一个稠密块的输出通道数
        num_channels += num_convs * growth_rate
        # 在稠密块之间添加一个转换层，使通道数量减半
        if i != len(num_convs_in_dense_blocks) - 1:
            blks.append(transition_block(num_channels, num_channels // 2))
            num_channels = num_channels // 2

    # 经过Dense模块，通道数为400，尺寸为14
    layer4 = nn.Sequential(*blks,
                           nn.BatchNorm2d(num_channels), nn.GELU(), nn.MaxPool2d(kernel_size=3, stride=1, padding=1))

    layer5 = nn.Sequential(*resnet_block(400, 400, 2, first_block=True))
    layer6 = nn.Sequential(*resnet_block(400, 800, 2))

    net = nn.Sequential(layer1, layer2, layer3, layer4, layer5, layer6, nn.AdaptiveAvgPool2d((1, 1)),
                         nn.Flatten(), nn.Linear(800, num_classes))

    return net


if __name__ == "__main__":
    X = torch.rand(size=(10, 3, 224, 224))
    model = model_B(num_classes=10)
    for layer in model:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)
