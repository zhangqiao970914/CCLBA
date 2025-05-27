import torch
from torch import nn
import torch.nn.functional as F
from models.Swin import Swintransformer
from util import *
from torch import nn, einsum
import numpy as np
from einops import rearrange, repeat


def weights_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.BatchNorm1d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.ReLU, Act, nn.AdaptiveAvgPool2d, nn.Softmax)):
            pass
        else:
            m.initialize()


def resize(input, target_size=(224, 224)):
    return F.interpolate(input, (target_size[0], target_size[1]), mode='bilinear', align_corners=True)


"""
minmax_norm:
    将tensor (shape=[N, 1, H, W]) 中的值拉伸到 [0, 1] 之间. 用于对输入的深度图进行初步处理.
"""


class CA(nn.Module):
    def __init__(self, channel):
        super(CA, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // 4, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SA(nn.Module):
    def __init__(self, kernel_size=3):
        super(SA, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


class CyclicShift(nn.Module):
    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def forward(self, x):
        return torch.roll(x, shifts=(self.displacement, self.displacement), dims=(1, 2))


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


def create_mask(window_size, displacement, upper_lower, left_right):
    mask = torch.zeros(window_size ** 2, window_size ** 2)

    if upper_lower:
        mask[-displacement * window_size:, :-displacement * window_size] = float('-inf')
        mask[:-displacement * window_size, -displacement * window_size:] = float('-inf')

    if left_right:
        mask = rearrange(mask, '(h1 w1) (h2 w2) -> h1 w1 h2 w2', h1=window_size, h2=window_size)
        mask[:, -displacement:, :, :-displacement] = float('-inf')
        mask[:, :-displacement, :, -displacement:] = float('-inf')
        mask = rearrange(mask, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')

    return mask


def get_relative_distances(window_size):
    indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
    distances = indices[None, :, :] - indices[:, None, :]
    return distances


class WindowAttention(nn.Module):
    def __init__(self, dim, heads, head_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        inner_dim = head_dim * heads

        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted

        if self.shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift(-displacement)
            self.cyclic_back_shift = CyclicShift(displacement)
            self.upper_lower_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                             upper_lower=True, left_right=False), requires_grad=False)
            self.left_right_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                            upper_lower=False, left_right=True), requires_grad=False)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(window_size) + window_size - 1
            self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_size ** 2, window_size ** 2))

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        if self.shifted:
            x = self.cyclic_shift(x)

        b, n_h, n_w, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        nw_h = n_h // self.window_size
        nw_w = n_w // self.window_size

        q, k, v = map(
            lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d',
                                h=h, w_h=self.window_size, w_w=self.window_size), qkv)

        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale

        if self.relative_pos_embedding:
            dots += self.pos_embedding[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]
        else:
            dots += self.pos_embedding

        if self.shifted:
            dots[:, :, -nw_w:] += self.upper_lower_mask
            dots[:, :, nw_w - 1::nw_w] += self.left_right_mask

        attn = dots.softmax(dim=-1)

        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)
        out = rearrange(out, 'b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)',
                        h=h, w_h=self.window_size, w_w=self.window_size, nw_h=nw_h, nw_w=nw_w)
        out = self.to_out(out)

        if self.shifted:
            out = self.cyclic_back_shift(out)
        return out


class SwinBlock(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        self.attention_block = WindowAttention(dim=2 * dim,
                                               heads=heads,
                                               head_dim=head_dim,
                                               shifted=shifted,
                                               window_size=window_size,
                                               relative_pos_embedding=relative_pos_embedding)
        self.mlp_block1 = FeedForward(dim=dim, hidden_dim=mlp_dim)
        self.mlp_block2 = FeedForward(dim=dim, hidden_dim=mlp_dim)


    def forward(self, x, d):
        x = self.norm1(x)
        d = self.norm2(d)

        xd = torch.cat([x, d], dim=-1)
        xd = self.attention_block(xd)

        x_sa, d_sa = torch.chunk(xd, 2, dim=-1)

        x_sa = x_sa + x
        d_sa = d_sa + d

        x_sa = self.norm3(x_sa)
        d_sa = self.norm4(d_sa)

        x_mlp = self.mlp_block1(x_sa) + x_sa
        d_mlp = self.mlp_block2(d_sa) + d_sa
        return x_mlp, d_mlp


class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, downscaling_factor):
        super().__init__()
        self.downscaling_factor = downscaling_factor
        self.patch_merge = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor, padding=0)
        self.linear = nn.Linear(in_channels * downscaling_factor ** 2, out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
        x = self.patch_merge(x).view(b, -1, new_h, new_w).permute(0, 2, 3, 1)
        x = self.linear(x)
        return x


class StageModule(nn.Module):
    def __init__(self, in_channels, hidden_dimension, layers, downscaling_factor, num_heads, head_dim, window_size,
                 relative_pos_embedding):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        self.patch_partition = PatchMerging(in_channels=in_channels, out_channels=hidden_dimension,
                                            downscaling_factor=downscaling_factor)

        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(nn.ModuleList([
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
            ]))

    def forward(self, x, d):
        x = self.patch_partition(x)
        d = self.patch_partition(d)
        for regular_block, shifted_block in self.layers:
            x, d = regular_block(x, d)
            x, d = shifted_block(x, d)
        return x.permute(0, 3, 1, 2), d.permute(0, 3, 1, 2)


def minmax_norm(pred):
    N, _, H, W = pred.shape
    pred = pred.view(N, -1)  # [N, HW]
    max_value = torch.max(pred, dim=1, keepdim=True)[0]  # [N, 1]
    min_value = torch.min(pred, dim=1, keepdim=True)[0]  # [N, 1]
    norm_pred = (pred - min_value) / (max_value - min_value + 1e-12)  # [N, HW]
    norm_pred = norm_pred.view(N, 1, H, W)
    return norm_pred


class EnLayer(nn.Module):
    def __init__(self, in_channel=64):
        super(EnLayer, self).__init__()
        self.enlayer = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.enlayer(x)
        return x


class LatLayer(nn.Module):
    def __init__(self, in_channel):
        super(LatLayer, self).__init__()
        self.convlayer = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.convlayer(x)
        return x


class DSLayer(nn.Module):
    def __init__(self, in_channel=64):
        super(DSLayer, self).__init__()
        self.enlayer = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.predlayer = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0))  # , nn.Sigmoid())

    def forward(self, x):
        x = self.enlayer(x)
        x = self.predlayer(x)
        return x


class EAB(nn.Module):
    def __init__(self, channel):
        super(EAB, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(channel, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv_atten = nn.Conv2d(2, 1, 3, padding=1, bias=False)
        self.edg_pred = nn.Conv2d(64, 1, 1)

    def forward(self, rd, g):
        rd = self.conv1(rd)
        avg_out = torch.mean(rd, dim=1, keepdim=True)
        max_out, _ = torch.max(rd, dim=1, keepdim=True)
        atten = torch.cat([avg_out, max_out], dim=1)
        atten = torch.sigmoid(self.conv_atten(atten))
        r_att = 1 - torch.sigmoid(atten)
        fuse = r_att * g
        return self.edg_pred(fuse)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.toplayer = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0))

        self.latlayer3 = LatLayer(in_channel=64)
        self.latlayer2 = LatLayer(in_channel=64)
        self.latlayer1 = LatLayer(in_channel=64)

        self.enlayer3 = EnLayer()
        self.enlayer2 = EnLayer()
        self.enlayer1 = EnLayer()

        self.dslayer3 = DSLayer()
        self.dslayer2 = DSLayer()
        self.dslayer1 = DSLayer()

        self.eab3 = EAB(64)
        self.eab2 = EAB(64)
        self.eab1 = EAB(64)

    def _upsample_add(self, x, y):
        [_, _, H, W] = y.size()
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        return x + y

    def forward(self, x4, d4, x3, d3, x2, d2, x1, d1, H, W):
        f4 = x4 + d4
        f3 = x3 + d3
        f2 = x2 + d2
        f1 = x1 + d1
        preds = []
        preds_edg = []
        p4 = self.toplayer(f4)
        p3 = self._upsample_add(p4, self.latlayer3(f3))
        p3 = self.enlayer3(p3)

        p3_edg = self.eab3(f3, p3)
        preds_edg.append(
            F.interpolate(p3_edg,
                          size=(H, W),
                          mode='bilinear', align_corners=False))
        _pred = self.dslayer3(p3)
        preds.append(
            F.interpolate(_pred,
                          size=(H, W),
                          mode='bilinear', align_corners=False))

        p2 = self._upsample_add(p3, self.latlayer2(f2))
        p2 = self.enlayer2(p2)

        p2_edg = self.eab2(f2, p2)
        preds_edg.append(
            F.interpolate(p2_edg,
                          size=(H, W),
                          mode='bilinear', align_corners=False))

        _pred = self.dslayer2(p2)
        preds.append(
            F.interpolate(_pred,
                          size=(H, W),
                          mode='bilinear', align_corners=False))

        p1 = self._upsample_add(p2, self.latlayer1(f1))
        p1 = self.enlayer1(p1)
        p1_edg = self.eab1(f1, p1)
        preds_edg.append(
            F.interpolate(p1_edg,
                          size=(H, W),
                          mode='bilinear', align_corners=False))
        _pred = self.dslayer1(p1)
        preds.append(
            F.interpolate(_pred,
                          size=(H, W),
                          mode='bilinear', align_corners=False))
        return preds, preds_edg


class CFM(nn.Module):
    def __init__(self):
        super(CFM, self).__init__()

        self.toplayer_r = nn.Sequential(
            nn.Conv2d(384, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0))
        self.latlayer3_r = LatLayer(in_channel=384)
        self.latlayer2_r = LatLayer(in_channel=192)
        self.latlayer1_r = LatLayer(in_channel=96)
        self.toplayer_d = nn.Sequential(
            nn.Conv2d(384, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0))
        self.latlayer3_d = LatLayer(in_channel=384)
        self.latlayer2_d = LatLayer(in_channel=192)
        self.latlayer1_d = LatLayer(in_channel=96)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.stage3 = StageModule(in_channels=64, hidden_dimension=64, layers=6,
                                  downscaling_factor=1, num_heads=8, head_dim=8,
                                  window_size=7, relative_pos_embedding=True)
        self.stage2 = StageModule(in_channels=64, hidden_dimension=64, layers=4,
                                  downscaling_factor=1, num_heads=8, head_dim=8,
                                  window_size=7, relative_pos_embedding=True)
        self.stage1 = StageModule(in_channels=64, hidden_dimension=64, layers=2,
                                  downscaling_factor=1, num_heads=8, head_dim=8,
                                  window_size=7, relative_pos_embedding=True)
        self.ca = CA(64)
        self.sa = SA()
    def _upsample_add(self, x, y):
        [_, _, H, W] = y.size()
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        return x + y
    def casa(self, x, d):
        xd = x * d
        xd_ca = self.ca(xd)
        x_ca = x * xd_ca
        d_ca = d * xd_ca

        x_sa = self.sa(x_ca) * x
        d_sa = self.sa(d_ca) * d

        x_out = x_sa + x
        d_out = d_sa + d
        return x_out, d_out
    def forward(self, x4, d4, mask_r, mask_d, x3, d3, x2, d2, x1, d1):
        x4 = self.toplayer_r(x4)
        d4 = self.toplayer_d(d4)

        x3 = self._upsample_add(x4, self.latlayer3_r(x3 * mask_r))
        d3 = self._upsample_add(d4, self.latlayer3_d(d3 * mask_d))
        x3, d3 = self.casa(x3, d3)
        x3, d3 = self.stage3(x3, d3)

        x2 = self._upsample_add(x3, self.latlayer2_r(x2 * self.upsample2(mask_r)))
        d2 = self._upsample_add(d3, self.latlayer2_d(d2 * self.upsample2(mask_d)))
        x2, d2 = self.casa(x2, d2)
        x2, d2 = self.stage2(x2, d2)

        x1 = self._upsample_add(x2, self.latlayer1_r(x1 * self.upsample4(mask_r)))
        d1 = self._upsample_add(d2, self.latlayer1_d(d1 * self.upsample4(mask_d)))
        x1, d1 = self.casa(x1, d1)
        x1, d1 = self.stage1(x1, d1)

        return x3, d3, x2, d2, x1, d1


class AttLayer(nn.Module):
    def __init__(self, input_channels=512):
        super(AttLayer, self).__init__()
        self.query_transform = nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0)
        self.key_transform = nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0)

        self.query_transform1 = nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0)
        self.key_transform1 = nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0)

        self.scale = 1.0 / (input_channels ** 0.5)
        self.conv_r = nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0)
        self.conv_d = nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0)

    def correlation(self, x5, seeds):
        B, C, H5, W5 = x5.size()
        if self.training:
            correlation_maps1 = F.conv2d(x5, weight=seeds, dilation=1)
            correlation_maps2 = F.conv2d(x5, weight=seeds, dilation=3)
            correlation_maps3 = F.conv2d(x5, weight=seeds, dilation=5)  # B,B,H,W
            correlation_maps = correlation_maps1 + correlation_maps2 + correlation_maps3            
        else:
            correlation_maps1 = F.conv2d(x5, weight=seeds, dilation=1)
            correlation_maps2 = F.conv2d(x5, weight=seeds, dilation=3)
            correlation_maps3 = F.conv2d(x5, weight=seeds, dilation=5)  # B,B,H,W
            correlation_maps = torch.relu(correlation_maps1 + correlation_maps2 + correlation_maps3)
        correlation_maps = correlation_maps.mean(1).view(B, -1)
        min_value = torch.min(correlation_maps, dim=1, keepdim=True)[0]
        max_value = torch.max(correlation_maps, dim=1, keepdim=True)[0]
        correlation_maps = (correlation_maps - min_value) / (max_value - min_value + 1e-12)  # shape=[B, HW]
        correlation_maps = correlation_maps.view(B, 1, H5, W5)  # shape=[B, 1, H, W]
        return correlation_maps

    def forward(self, x5, d5):
        # x: B,C,H,W
        x5 = self.conv_r(x5) + x5 + d5
        d5 = self.conv_d(d5) + d5 + x5
        ##RGB##
        B, C, H5, W5 = x5.size()
        x_query = self.query_transform(x5).view(B, C, -1)
        x_query = torch.transpose(x_query, 1, 2).contiguous().view(-1, C)  # BHW, C
        x_key = self.key_transform(x5).view(B, C, -1)
        x_key = torch.transpose(x_key, 0, 1).contiguous().view(C, -1)  # C, BHW

        ##Depth##
        d_query = self.query_transform1(d5).view(B, C, -1)
        d_query = torch.transpose(d_query, 1, 2).contiguous().view(-1, C)  # BHW, C
        d_key = self.key_transform1(d5).view(B, C, -1)
        d_key = torch.transpose(d_key, 0, 1).contiguous().view(C, -1)  # C, BHW

        x_w1 = torch.matmul(x_query, x_key) * self.scale  # BHW, BHW
        d_w1 = torch.matmul(d_query, d_key) * self.scale  # BHW, BHW

        xd_w1 = torch.matmul(x_query, d_key) * self.scale  # BHW, BHW
        dx_w1 = torch.matmul(d_query, x_key) * self.scale  # BHW, BHW

        f_xw = x_w1 + d_w1 + xd_w1 + dx_w1
        f_xw = f_xw.view(B * H5 * W5, B, H5 * W5)  # [BHW, B, HW]
        f_xw_max = torch.max(f_xw, dim=-1)[0]  # [BHW, B]
        f_xw_max_avg = f_xw_max.mean(-1)  # [BHW, B]
        c_co = f_xw_max_avg.view(B, -1)
        c_co = F.softmax(c_co, dim=-1)  # [B,HW]

        ###fuse###
        fuse = c_co

        norm_r = F.normalize(x5, dim=1)
        norm_d = F.normalize(d5, dim=1)

        fuse = fuse.unsqueeze(1)
        fuse_max = torch.max(fuse, -1).values.unsqueeze(2).expand_as(fuse)
        mask = torch.zeros_like(fuse).cuda()
        mask[fuse == fuse_max] = 1
        mask = mask.view(B, 1, H5, W5)
        seeds_r = norm_r * mask
        seeds_d = norm_d * mask

        seeds_r = seeds_r.sum(3).sum(2).unsqueeze(2).unsqueeze(3)
        seeds_d = seeds_d.sum(3).sum(2).unsqueeze(2).unsqueeze(3)
        cormap_r = self.correlation(norm_r, seeds_r)
        cormap_d = self.correlation(norm_d, seeds_d)

        x51 = x5 * cormap_r
        d51 = d5 * cormap_d
        proto_r = torch.mean(x51, (0, 2, 3), True)
        proto_d = torch.mean(d51, (0, 2, 3), True)

        final_r = x5 * proto_r + x51
        final_d = d5 * proto_d + d51
        return final_r, final_d, cormap_r, cormap_d


class DCFMNet(nn.Module):
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, mode='train'):
        super(DCFMNet, self).__init__()
        self.gradients = None
        self.backbone = Swintransformer(224)
        self.backbone.load_state_dict(
            torch.load('/hy-tmp/DCFM-CoSOD_Depth/models/swin_base_patch4_window7_224_22k.pth')['model'], strict=False)
        self.mode = mode
        self.mcm = AttLayer(384)
        self.cfm = CFM()

        self.decoder = Decoder()

    def set_mode(self, mode):
        self.mode = mode

    def forward(self, x, dep, gt):
        if self.mode == 'train':
            preds = self._forward(x, dep, gt)
        else:
            with torch.no_grad():
                preds = self._forward(x, dep, gt)

        return preds

    def featextract(self, x, d):
        x1, x2, x3, x4 = self.backbone(x)
        d1, d2, d3, d4 = self.backbone(d)
        return x4, x3, x2, x1, d4, d3, d2, d1

    def _forward(self, x, dep, gt):
        [B, _, H, W] = x.size()

        dep = minmax_norm(dep).repeat(1, 3, 1, 1)
        x4, x3, x2, x1, d4, d3, d2, d1 = self.featextract(x, dep)
        x4, d4, mask_r, mask_d = self.mcm(x4, d4)
        x3, d3, x2, d2, x1, d1 = self.cfm(x4, d4, mask_r, mask_d, x3, d3, x2, d2, x1, d1)
        preds, preds_edg = self.decoder(x4, d4, x3, d3, x2, d2, x1, d1, H, W)
        if self.training:
            return preds, preds_edg
        return preds


class DCFM(nn.Module):
    def __init__(self, mode='train'):
        super(DCFM, self).__init__()
        set_seed(123)
        self.dcfmnet = DCFMNet()
        self.mode = mode

    def set_mode(self, mode):
        self.mode = mode
        self.dcfmnet.set_mode(self.mode)

    def forward(self, x, dep, gt):
        ########## Co-SOD ############
        preds = self.dcfmnet(x, dep, gt)
        return preds

