import torch
import torch.nn as nn


# def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
# #     layer = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias, stride=stride)
# #     return layer
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False, stride=1, norm=False):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias, stride=stride)
        self.norm = norm

        if self.norm == 'BN':
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif self.norm == 'LN':
            self.norm_layer = nn.InstanceNorm2d(num_features=out_channels, momentum=0.3, affine=True, track_running_stats=True)

    def forward(self, x):
        if self.norm:
            return self.norm_layer(self.conv(x))
        else:
            return self.conv(x)


class Conv_Relu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Conv_Relu, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))

    def forward(self, x):
        return self.conv(x)

class DeConv_Relu(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeConv_Relu, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))

    def forward(self, x):
        return self.deconv(x)

class SALayer(nn.Module):
    def __init__(self, kernel_size=7):
        super(SALayer, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv1(y)
        y = self.sigmoid(y)
        return x * y

# Spatial Attention Block (SAB)
class SAB(nn.Module):
    def __init__(self, n_feat, kernel_size, bias, act):
        super(SAB, self).__init__()
        modules_body = [Conv(n_feat, n_feat, kernel_size, bias=bias), act, Conv(n_feat, n_feat, kernel_size, bias=bias)]
        self.body = nn.Sequential(*modules_body)
        self.SA = SALayer(kernel_size=7)

    def forward(self, x):
        res = self.body(x)
        res = self.SA(res)
        res += x
        return res

# Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.avg_pool = nn.AdaptiveMaxPool2d(1)

        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


# Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = [Conv(n_feat, n_feat, kernel_size, bias=bias), act, Conv(n_feat, n_feat, kernel_size, bias=bias)]

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res

## Pixel Attention
class PALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),  # channel <-> 1
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


## Pixel Attention Block (PAB)
class PAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(PAB, self).__init__()
        modules_body = [Conv(n_feat, n_feat, kernel_size, bias=bias), act, Conv(n_feat, n_feat, kernel_size, bias=bias)]
        self.PA = PALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.PA(res)
        res += x
        return res

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False),
                                nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bicubic', align_corners=False),
                                nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x

class CBAM(nn.Module):
    def __init__(self, n_feats, kernel_size=3, reduction=4, bias=False, act=nn.ReLU(inplace=True)):
        super(CBAM, self).__init__()
        self.channel_attention = CAB(n_feats, kernel_size, reduction, bias, act)
        self.spatial_attention = SAB(n_feats, kernel_size, bias, act)

    def forward(self, x):
        out = self.channel_attention(x)
        out = self.spatial_attention(out)
        res = out + x
        return res