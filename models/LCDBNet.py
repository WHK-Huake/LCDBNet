import torch
import torch.nn as nn
import torch.nn.functional as TF
from torch.autograd import Variable
import utils
from models.PixelUnShuffle import DWT, IWT, PixelUnShuffle
from models.common import Conv, CAB, DownSample, UpSample, CALayer, SALayer

# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import numpy as np
from thop import profile
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from timm.models.layers import trunc_normal_, DropPath


class WMSA(nn.Module):
    """ Self-attention module in Swin Transformer
    """

    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super(WMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim // head_dim
        self.window_size = window_size
        self.type = type
        self.embedding_layer = nn.Linear(self.input_dim, 3 * self.input_dim, bias=True)

        # TODO recover
        # self.relative_position_params = nn.Parameter(torch.zeros(self.n_heads, 2 * window_size - 1, 2 * window_size -1))
        self.relative_position_params = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), self.n_heads))

        self.linear = nn.Linear(self.input_dim, self.output_dim)

        trunc_normal_(self.relative_position_params, std=.02)
        self.relative_position_params = torch.nn.Parameter(
            self.relative_position_params.view(2 * window_size - 1, 2 * window_size - 1, self.n_heads).transpose(1,
                                                                                                                 2).transpose(
                0, 1))

    def generate_mask(self, h, w, p, shift):
        """ generating the mask of SW-MSA
        Args:
            shift: shift parameters in CyclicShift.
        Returns:
            attn_mask: should be (1 1 w p p),
        """
        # supporting sqaure.
        attn_mask = torch.zeros(h, w, p, p, p, p, dtype=torch.bool, device=self.relative_position_params.device)
        if self.type == 'W':
            return attn_mask

        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        attn_mask = rearrange(attn_mask, 'w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)')
        return attn_mask

    def forward(self, x):
        """ Forward pass of Window Multi-head Self-attention module.
        Args:
            x: input tensor with shape of [b h w c];
            attn_mask: attention mask, fill -inf where the value is True;
        Returns:
            output: tensor shape [b h w c]
        """
        if self.type != 'W': x = torch.roll(x, shifts=(-(self.window_size // 2), -(self.window_size // 2)), dims=(1, 2))
        x = rearrange(x, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)
        h_windows = x.size(1)
        w_windows = x.size(2)
        # sqaure validation
        # assert h_windows == w_windows

        x = rearrange(x, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        qkv = self.embedding_layer(x)
        q, k, v = rearrange(qkv, 'b nw np (threeh c) -> threeh b nw np c', c=self.head_dim).chunk(3, dim=0)
        sim = torch.einsum('hbwpc,hbwqc->hbwpq', q, k) * self.scale
        # Adding learnable relative embedding
        sim = sim + rearrange(self.relative_embedding(), 'h p q -> h 1 1 p q')
        # Using Attn Mask to distinguish different subwindows.
        if self.type != 'W':
            attn_mask = self.generate_mask(h_windows, w_windows, self.window_size, shift=self.window_size // 2)
            sim = sim.masked_fill_(attn_mask, float("-inf"))

        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum('hbwij,hbwjc->hbwic', probs, v)
        output = rearrange(output, 'h b w p c -> b w p (h c)')
        output = self.linear(output)
        output = rearrange(output, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c', w1=h_windows, p1=self.window_size)

        if self.type != 'W': output = torch.roll(output, shifts=(self.window_size // 2, self.window_size // 2),
                                                 dims=(1, 2))
        return output

    def relative_embedding(self):
        cord = torch.tensor(np.array([[i, j] for i in range(self.window_size) for j in range(self.window_size)]))
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size - 1
        # negative is allowed
        return self.relative_position_params[:, relation[:, :, 0].long(), relation[:, :, 1].long()]


class TransBlock(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, type='W', input_resolution=None):
        """ SwinTransformer Block
        """
        super(TransBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ['W', 'SW']
        self.type = type
        if input_resolution <= window_size:
            self.type = 'W'

        print("TransBlock Initial Type: {}, drop_path_rate:{:.6f}".format(self.type, drop_path))
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.type)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )

    def forward(self, x):
        x = x + self.drop_path(self.msa(self.ln1(x)))
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x


class AttenBlock(nn.Module):
    def __init__(self, dim):
        super(AttenBlock, self).__init__()
        self.conv_first = Conv(dim, dim, kernel_size=3, bias=False, stride=1, norm=False)
        self.ca = CALayer(channel=dim, reduction=16, bias=False)
        self.sa = SALayer(kernel_size=7)
        self.conv = Conv(dim, dim, kernel_size=3, bias=False, stride=1, norm=False)
    def forward(self, x):
        x_conv = self.conv_first(x)
        x_ca = self.ca(x_conv)
        x_sa = self.sa(x_conv)
        x_out = self.conv(x_ca + x_sa)
        return x_out


class ConvTransBlock(nn.Module):
    def __init__(self, conv_dim, trans_dim, head_dim, window_size, drop_path, type='W', input_resolution=None):
        """ SwinTransformer and Conv Block
        """
        super(ConvTransBlock, self).__init__()
        self.conv_dim = conv_dim
        self.trans_dim = trans_dim
        self.head_dim = head_dim
        self.window_size = window_size
        self.drop_path = drop_path
        self.type = type
        self.input_resolution = input_resolution

        assert self.type in ['W', 'SW']
        if self.input_resolution <= self.window_size:
            self.type = 'W'

        self.trans_block = TransBlock(self.trans_dim, self.trans_dim, self.head_dim, self.window_size, self.drop_path,
                                 self.type, self.input_resolution)
        self.conv1_1 = nn.Conv2d(self.conv_dim + self.trans_dim, self.conv_dim + self.trans_dim, 1, 1, 0, bias=True)

        self.conv1_2 = nn.Sequential(CAB(self.conv_dim + self.trans_dim, kernel_size=3, reduction=8, bias=False, act=nn.ReLU(True)),
                                     nn.Conv2d(self.conv_dim + self.trans_dim, self.conv_dim + self.trans_dim, 1, 1, 0, bias=True))

        self.conv_block = AttenBlock(dim=self.conv_dim)


    def forward(self, x):
        conv_x, trans_x = torch.split(self.conv1_1(x), (self.conv_dim, self.trans_dim), dim=1)
        conv_x = self.conv_block(conv_x) + conv_x
        trans_x = Rearrange('b c h w -> b h w c')(trans_x)
        trans_x = self.trans_block(trans_x)
        trans_x = Rearrange('b h w c -> b c h w')(trans_x)
        res = self.conv1_2(torch.cat((conv_x, trans_x), dim=1))
        x = x + res

        return x


class LAN(nn.Module):

    def __init__(self, in_nc=3, dim=32, out_nc=64, config=[2, 2, 2, 2, 2], drop_path_rate=0.0, input_resolution=128):
        super(LAN, self).__init__()
        self.config = config
        self.dim = dim
        self.head_dim = 16
        self.window_size = 8

        # drop path rate for each layer
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]

        self.m_head = [nn.Conv2d(in_nc, dim, 3, 1, 1, bias=False)]

        begin = 0
        self.m_down1 = [ConvTransBlock(dim // 2, dim // 2, self.head_dim, self.window_size, dpr[i + begin],
                                       'W' if not i % 2 else 'SW', input_resolution)
                        for i in range(config[0])] + [nn.Conv2d(dim, 2 * dim, 2, 2, 0, bias=False)]

        begin += config[0]
        self.m_down2 = [ConvTransBlock(dim, dim, self.head_dim, self.window_size, dpr[i + begin],
                                       'W' if not i % 2 else 'SW', input_resolution // 2)
                        for i in range(config[1])] + [nn.Conv2d(2 * dim, 4 * dim, 2, 2, 0, bias=False)]

        begin += config[1]

        self.m_body = [ConvTransBlock(2 * dim, 2 * dim, self.head_dim, self.window_size, dpr[i + begin],
                                      'W' if not i % 2 else 'SW', input_resolution // 4)
                       for i in range(config[2])]

        begin += config[2]
        self.m_up2 = [nn.ConvTranspose2d(4 * dim, 2 * dim, 2, 2, 0, bias=False), ] + \
                     [ConvTransBlock(dim, dim, self.head_dim, self.window_size, dpr[i + begin],
                                     'W' if not i % 2 else 'SW', input_resolution // 2)
                      for i in range(config[3])]

        begin += config[3]
        self.m_up1 = [nn.ConvTranspose2d(2 * dim, dim, 2, 2, 0, bias=False), ] + \
                     [ConvTransBlock(dim // 2, dim // 2, self.head_dim, self.window_size, dpr[i + begin],
                                     'W' if not i % 2 else 'SW', input_resolution)
                      for i in range(config[4])]

        self.sam = SM(in_nc, dim, kernel_size=3, bias=False)
        self.tail_out = Conv(dim, out_nc, 3)

        self.m_head = nn.Sequential(*self.m_head)
        self.m_down1 = nn.Sequential(*self.m_down1)
        self.m_down2 = nn.Sequential(*self.m_down2)

        self.m_body = nn.Sequential(*self.m_body)
        self.m_up2 = nn.Sequential(*self.m_up2)
        self.m_up1 = nn.Sequential(*self.m_up1)

    def forward(self, x0):
        h, w = x0.size()[-2:]
        paddingBottom = int(np.ceil(h / 64) * 64 - h)
        paddingRight = int(np.ceil(w / 64) * 64 - w)
        x0 = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x0)

        x1 = self.m_head(x0)    # [1, 32, 128, 128]
        x2 = self.m_down1(x1)   # [1, 64, 64, 64]
        x3 = self.m_down2(x2)   # [1, 128, 32, 32]

        x = self.m_body(x3)     # [1, 128, 32, 32]
        x = self.m_up2(x + x3)  # [1, 64, 64, 64]
        x = self.m_up1(x + x2)  # [1, 32, 128, 128]

        x_feats, x_out = self.sam(x, x0)
        x_feats = self.tail_out(x_feats)
        x_feats = x_feats[..., :h, :w]
        x_out = x_out[..., :h, :w]

        return x_out, x_feats

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class MCAB(nn.Module):
    def __init__(self, in_feat, out_feat, kernel_size, reduction, n_blocks, bias=False, act=nn.ReLU(True)):
        super(MCAB, self).__init__()
        self.conv1 = Conv(in_feat, out_feat, 3)
        self.cab = nn.Sequential(*[CAB(out_feat, kernel_size=kernel_size, reduction=reduction, bias=bias, act=act) for _ in range(n_blocks)])
        self.conv2 = Conv(out_feat, out_feat, 3)

    def forward(self, x):
        x = self.conv1(x)
        x1 = self.cab(x) + x
        x = self.conv2(x1)
        return x


class SM(nn.Module):
    def __init__(self, channels, n_feat, kernel_size, bias):
        super(SM, self).__init__()
        self.conv1 = Conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = Conv(n_feat, channels, kernel_size, bias=bias)
        self.conv3 = Conv(channels, n_feat, kernel_size, bias=bias)
        self.conv4 = Conv(channels, n_feat, kernel_size, bias=bias)
    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1 * x2
        x1 = x1 + x
        x0 = self.conv4(x_img)
        x1 = x1 + x0
        return x1, img

class CRN(nn.Module):
    def __init__(self,  channels, features=64, out_chans=64):
        super(CRN, self).__init__()
        self.dwt = DWT()
        self.iwt = IWT()

        self.layer1 = MCAB(in_feat=channels, out_feat=features, kernel_size=3, reduction=4, n_blocks=3, bias=False, act=nn.ReLU(True))
        self.layer2 = MCAB(in_feat=channels * 4, out_feat=features * 4, kernel_size=3, reduction=4, n_blocks=3, bias=False, act=nn.ReLU(True))
        self.layer3 = MCAB(in_feat=channels * 16, out_feat=features * 8, kernel_size=3, reduction=4, n_blocks=3,  bias=False, act=nn.ReLU(True))
        self.fuse1 = Conv(features * 2 + features // 2, features, 3)

        self.layer4 = MCAB(in_feat=features, out_feat=features, kernel_size=3, reduction=4, n_blocks=3, bias=False, act=nn.ReLU(True))
        self.sbm = SM(channels, features, 3, bias=False)
        self.out = Conv(features, out_chans, 3)

    def forward(self, x):
        x1 = x
        x2 = self.dwt(x1)
        x3 = self.dwt(x2)

        x1 = self.layer1(x1)        # C
        x2 = self.iwt(self.layer2(x2))      # C
        x3 = self.iwt(self.iwt(self.layer3(x3)))

        x_cat = torch.cat([x1, x2, x3], 1)
        x_cat = self.fuse1(x_cat)
        x_cat = self.layer4(x_cat)
        feats, out = self.sbm(x_cat, x)
        feats = self.out(feats)
        return out, feats


class FN(nn.Module):
    def __init__(self, features):
        super(FN, self).__init__()
        layers = []
        for _ in range(5):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False))
            layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)
        self.out = Conv(features, 3, 3)

    def forward(self, x):
        return self.out(self.layers(x))


class LCDBNet(nn.Module):
    def __init__(self):
        super(LCDBNet, self).__init__()

        self.illum = LAN(in_nc=1, dim=32, out_nc=48)
        self.denoise = CRN(channels=2, features=32, out_chans=48)
        self.mix = FN(features=96)

    def forward(self, x):

        y = x[:, 0, :, :].unsqueeze(1)
        uv = x[:, 1:3, :, :]

        out_y, out_y_feats = self.illum(y)
        out_uv, out_uv_feats = self.denoise(uv)

        feats = torch.cat([out_y_feats, out_uv_feats], 1)
        out = self.mix(feats)

        return out_y, out_uv, out

