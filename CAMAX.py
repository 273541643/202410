class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.pool_max_h = nn.AdaptiveMaxPool2d((None, 1))  # 添加自适应最大池化层
        self.pool_max_w = nn.AdaptiveMaxPool2d((1, None))  # 添加自适应最大池化层

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_max_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)  # 添加Conv2d层
        self.conv_max_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)  # 添加Conv2d层

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()

        # 1. 自适应池化
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        max_x_h = self.pool_max_h(x)
        max_x_w = self.pool_max_w(x).permute(0, 1, 3, 2)

        # 2. 拼接操作
        y = torch.cat([x_h, x_w, max_x_h, max_x_w], dim=2)

        # 3. 中间变量计算
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # 4. 宽度和高度注意力系数的计算
        x_h, x_w, max_x_h, max_x_w = torch.split(y, [h, w, h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        max_x_w = max_x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        max_a_h = self.conv_max_h(max_x_h).sigmoid()
        max_a_w = self.conv_max_w(max_x_w).sigmoid()

        # 5. 最终输出元素值
        out = identity * a_w * a_h * max_a_w * max_a_h

        return out
