from torch.nn.modules import *
import torch
import torch.nn.functional as F


class LPA(Module):
    def __init__(self, n_feats):
        super(LPA, self).__init__()
        f = n_feats // 4
        self.collapse = conv(n_feats, f, 1)
        self.squeeze = MaxPool2d(4, 4)
        self.redistribution = conv(f, f, 3)
        self.restore = conv(f, n_feats, 1)
        self.sigmoid = Sigmoid()
        self.act = LeakyReLU(0.2, True)

    def forward(self, x):
        y = self.collapse(x)
        res = y
        y = self.squeeze(y)
        y = self.redistribution(y)
        y = F.interpolate(y, x.size()[2:], mode='bilinear', align_corners=False)
        y = self.restore(y + res)
        y = self.sigmoid(y)
        return x * y


def conv(in_channels, out_channels, kernel_size, stride=1, dilation=1):
    padding = kernel_size // 2 if dilation == 1 else dilation
    return Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)


class CFB(Module):
    def __init__(self, n_feats):
        super(CFB, self).__init__()
        self.conv1 = conv(n_feats, n_feats, 3)
        self.conv2 = conv(n_feats, n_feats, 3)
        self.conv3 = conv(n_feats, n_feats, 3)
        self.fusion = conv(n_feats * 3, n_feats, 1)
        self.act = LeakyReLU(0.2, True)
        self.att = LPA(n_feats)

    def forward(self, x):
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(x1 + x))
        x3 = self.act(self.conv3(x2 + x))
        x = self.att(self.fusion(torch.cat([x1, x2, x3], 1)) + x)
        return x


class CFNet(Module):
    def __init__(self, n_colors=3, n_feats=48, n_blocks=6, scale=2):
        super(CFNet, self).__init__()
        self.head = conv(n_colors, n_feats, 3)
        self.body = Sequential(*[CFB(n_feats) for _ in range(n_blocks)])
        self.fusion = Sequential(conv(n_feats * n_blocks, n_feats, 1), LPA(n_feats))
        self.tail = Sequential(conv(n_feats, n_colors * scale ** 2, 3),
                               PixelShuffle(scale))

    def forward(self, x):
        x = self.head(x)
        res, outputs = x, []
        for block in self.body:
            res = block(res)
            outputs.append(res)
        x = self.fusion(torch.cat(outputs, 1)) + x
        x = self.tail(x)
        return x

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, torch.nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
