import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self, c1, c2, k, s=1, p=0, d=1, g=1, act=True):
        super(Conv, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g),
            nn.BatchNorm2d(c2),
            nn.LeakyReLU(0.1, inplace=True) if act else nn.Identity()
        )

    def forward(self, x):
        return self.convs(x)


class SPP(nn.Module):
    """
        Spatial Pyramid Pooling
        SPP接受的输入特征图大小是13x13x512，
        经过四个maxpooling分支处理后，再汇合到一起，那么得到的是一个13x13x2048的特征图，代码中只用了三个
        这里，我们会再接一个1x1的卷积层（conv1x1+BN+LeakyReLU）将通道压缩一下，
        最终Neck部分的输入同样是13x13x512的特征图

    """
    def __init__(self):
        super(SPP, self).__init__()

    def forward(self, x):

        x_1 = torch.nn.functional.max_pool2d(x, 5, stride=1, padding=2)  # 都能保证shape不变 1 3 416 416
        x_2 = torch.nn.functional.max_pool2d(x, 9, stride=1, padding=4)
        x_3 = torch.nn.functional.max_pool2d(x, 13, stride=1, padding=6)
        x = torch.cat([x, x_1, x_2, x_3], dim=1)
        # 函数目的： 在给定维度上对输入的张量序列seq
        # 进行连接操作。

        return x


# spp_test = SPP()
# print(spp_test.forward(torch.randn(1, 3, 416, 416)).shape)
# 保证每次池化后的shape是不变的 所以用不同的size池化的时候，要加上padding
# 都是1 3 416 416 再按照 1  轴堆叠
# spp之后的张量： torch.Size([1, 12, 416, 416])
