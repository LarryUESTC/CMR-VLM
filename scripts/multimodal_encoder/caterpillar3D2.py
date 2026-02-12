import torch
from torch import nn
import torch.nn.functional as F
import os
# from ._helpers import clean_state_dict

from timm.models.vision_transformer import Mlp
# from ._registry import register_model
from timm.models.layers import DropPath, PatchEmbed
from .patch3D import PatchEmbeddingBlock
# from typing import Literal
# from thop import profile
# from thop import clever_format
# from vit import ViT
class ShiftedPillarsConcatentation_BNAct(nn.Module):

    def __init__(self, channels, step, D, H, W):
        super().__init__()
        self.channels = channels
        self.step = step
        # self.BN = nn.BatchNorm2d(channels)
        self.BN = nn.LayerNorm([channels, H, W])
        self.Act = nn.GELU()

        self.proj_t = nn.Conv2d(channels, channels // 4, (1, 1))
        self.proj_b = nn.Conv2d(channels, channels // 4, (1, 1))
        self.proj_l = nn.Conv2d(channels, channels // 4, (1, 1))
        self.proj_r = nn.Conv2d(channels, channels // 4, (1, 1))
        self.fuse = nn.Conv2d(channels, channels, (1, 1))

    def forward(self, x):
        B, C, D, H, W = x.shape
        new_shape = (B * D, C, H, W)
        x = x.permute(0,2,1,3,4).contiguous().reshape(*new_shape)
        x = self.Act(self.BN(x))

        x_t, x_b, x_r, x_l = x.clone(), x.clone(), x.clone(), x.clone()
        x_t = torch.narrow(x_t, 2, self.step, (H-self.step))
        x_b = torch.narrow(x_b, 2, 0, (H-self.step))
        x_l = torch.narrow(x_l, 3, self.step, (W-self.step))
        x_r = torch.narrow(x_r, 3, 0, (W-self.step))
        x_t = F.pad(x_t, (0, 0, 0, self.step), "constant", 0)
        x_b = F.pad(x_b, (0, 0, self.step, 0), "constant", 0)
        x_l = F.pad(x_l, (0, self.step, 0, 0), "constant", 0)
        x_r = F.pad(x_r, (self.step, 0, 0, 0), "constant", 0)

        x_t = self.proj_t(x_t)
        x_b = self.proj_b(x_b)
        x_l = self.proj_l(x_l)
        x_r = self.proj_r(x_r)

        x = self.fuse(torch.cat([x_t, x_b, x_r, x_l], dim=1))
        org_shape = (B, D, C, H, W)
        x = x.reshape(*org_shape).permute(0,2,1,3,4).contiguous()
        return x


class sparseMLP_BNAct(nn.Module):

    def __init__(self, D, H, W, channels):
        super().__init__()
        assert W == H
        self.channels = channels
        # self.BN = nn.BatchNorm3d(channels)
        self.BN = nn.LayerNorm([channels,D,H,W])
        self.Act = nn.GELU()
        self.W = W
        self.H = H
        self.D = D
        self.proj_h = nn.Conv3d(H, H, (1, 1, 1))
        self.proj_w = nn.Conv3d(W, W, (1, 1, 1))
        self.proj_d = nn.Conv3d(D, D, (1, 1, 1))
        self.fuse = nn.Conv3d(channels*4, channels, (1,1,1), (1,1,1), bias=False)

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = self.Act(self.BN(x))

        x_o, x_h, x_w, x_d = x.clone(), x.clone(), x.clone(), x.clone()
        x_d = self.proj_d(x_d.permute(0, 2, 1, 3, 4).contiguous()).permute(0, 2, 1, 3, 4).contiguous()
        x_h = self.proj_h(x_h.permute(0, 3, 2, 1, 4).contiguous()).permute(0, 3, 2, 1, 4).contiguous()
        x_w = self.proj_w(x_w.permute(0, 4, 2, 3, 1).contiguous()).permute(0, 4, 2, 3, 1).contiguous()
        x = self.fuse(torch.cat([x_o, x_d, x_h, x_w], dim=1))
        return x


class FeedForward_LN(nn.Module):

    def __init__(self, channels, drop_out=0.):
        super().__init__()
        self.LN = nn.LayerNorm(channels)
        self.FeedForward = Mlp(channels, channels * 4, channels, drop=drop_out)

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = self.LN(x)
        x = self.FeedForward(x)
        x = x.permute(0,4,1,2,3).contiguous()
        return x


class CaterpillarBlock(nn.Module):

    def __init__(self, D, H, W, channels, step, drop_out, drop_path):
        super().__init__()
        self.localMix = ShiftedPillarsConcatentation_BNAct(channels, step, D, H, W)
        self.globalMix = sparseMLP_BNAct(D, H, W, channels)
        self.channelMix = FeedForward_LN(channels,drop_out)

        self.drop_path_g = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path_c = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = self.drop_path_g(self.globalMix(self.localMix(x))) + x
        x = self.drop_path_c(self.channelMix(x)) + x
        return x


class Caterpillar(nn.Module):
    def __init__(self, img_size=224, in_chans=3, patch_size=4, embed_dim=[96,192,384,768], depth=[2,6,14,2],
                 down_stride=[2,2,2,2], shift_step=[1,1,1,1],
                 num_classes=1000, drop_rate=0., drop_path_rate=0.0, Patch_layer=PatchEmbed, act_layer=nn.GELU):
        super().__init__()
        assert patch_size == down_stride[0], "The down_sampling_stride[0] must be equal to the patch size."
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim[-1]


        self.patch_embed = Patch_layer(img_size=img_size, patch_size=patch_size,
                                     in_chans=in_chans, embed_dim=embed_dim[0], flatten=False, bias=False)
        num_patch = self.patch_embed.grid_size[0]
        dpr = [drop_path_rate * j / (sum(depth)) for j in range(sum(depth))]

        self.blocks = nn.ModuleList([])
        shift = 0
        for i in range(len(depth)):
            if i == 0:
                self.blocks.append(nn.Identity())
            else:
                num_patch = num_patch // down_stride[i]
                self.blocks.append(nn.Conv2d(embed_dim[i-1], embed_dim[i], down_stride[i], down_stride[i], bias=False))

            for j in range(depth[i]):
                drop_path=dpr[j+shift]
                self.blocks.append(nn.Sequential(CaterpillarBlock(num_patch, num_patch, embed_dim[i], shift_step[i],
                                                                  drop_out=drop_rate, drop_path=dpr[j+shift])))
            shift += depth[i]
            self.blocks = nn.Sequential(*self.blocks)

        self.norm = nn.BatchNorm2d(embed_dim[-1])


        self.feature_info = [dict(num_chs=embed_dim[-1], reduction=0, module='head')]
        self.head = nn.Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()


    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
        x = self.norm(x).mean(dim=[2,3]).flatten(1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

class Caterpillar3D(nn.Module):
    def __init__(self, img_size=[32, 256, 256], in_chans=1, patch_size=[4, 16, 16], embed_dim=[96,192,384,768], depth=[2,6,14,2],
                 down_stride=[2,2,2,2], shift_step=[1,1,1,1],
                 num_classes=1000, drop_rate=0., drop_path_rate=0.0, Patch_layer=PatchEmbed, act_layer=nn.GELU):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim[-1]


        self.patch_embed = PatchEmbeddingBlock(
            in_channels=in_chans,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=embed_dim[0],
            num_heads=4,
            proj_type="perceptron",
            dropout_rate=0.0,
            spatial_dims=3,
        )
        grid_size = self.patch_embed.grid_size
        dpr = [drop_path_rate * j / (sum(depth)) for j in range(sum(depth))]

        self.blocks = nn.ModuleList([])
        shift = 0
        for i in range(len(depth)):
            if i == 0:
                self.blocks.append(nn.Identity())
            else:
                grid_size = [grid_size[0] // 2, grid_size[1] // down_stride[i], grid_size[1] // down_stride[i]]
                self.blocks.append(nn.Conv3d(embed_dim[i-1], embed_dim[i], (2, down_stride[i], down_stride[i]), stride=(2, down_stride[i], down_stride[i]), bias=False))

            for j in range(depth[i]):
                # drop_path=dpr[j+shift]
                self.blocks.append(nn.Sequential(CaterpillarBlock(grid_size[0], grid_size[1], grid_size[2],
                                                                  embed_dim[i], shift_step[i],
                                                                  drop_out=drop_rate, drop_path=dpr[j+shift])))
            shift += depth[i]
            self.blocks = nn.Sequential(*self.blocks)

        # self.norm = nn.BatchNorm3d(embed_dim[-1])
        self.norm = nn.LayerNorm([embed_dim[-1],grid_size[0], grid_size[1], grid_size[2]])

        self.feature_info = [dict(num_chs=embed_dim[-1], reduction=0, module='head')]
        self.head = nn.Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()


    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        B, D_H_W, C = x.shape
        D, H, W = self.patch_embed.grid_size[0], self.patch_embed.grid_size[1], self.patch_embed.grid_size[2]
        assert D_H_W == D*H*W
        new_shape = (B, C, D, H, W)
        x = x.permute(0,2,1).contiguous()
        x = x.reshape(*new_shape)
        for i, blk in enumerate(self.blocks):
            x = blk(x)

        x = self.norm(x)
        x = x.permute(0,2,3,4,1).contiguous()
        x_cls = x.mean(dim=[1,2,3]).unsqueeze(1)
        B, D, H, W, C = x.shape
        out_shape = (B, D * H * W, C)
        x = x.reshape(*out_shape)
        return torch.cat((x_cls, x), dim=1), [x]

    def forward(self, x):
        # x = self.head(x)
        return self.forward_features(x)

# -------------Models for ImageNet-----------------

def Caterpillar_3D(pretrained=False, **kwargs):
    model = Caterpillar3D(img_size=[32, 256, 256], patch_size=[2, 4, 4],
                        embed_dim=[40,160,512,1024],
                        depth=[2,4,8,2],
                        down_stride=[4,2,2,2],
                        shift_step=[1,1,1,1], **kwargs)
    return model

def Caterpillar_Mi_IN1k(pretrained=False, **kwargs):
    model = Caterpillar(img_size=224, patch_size=4,
                        embed_dim=[40,80,160,320],
                        depth=[2,6,10,2],
                        down_stride=[4,2,2,2],
                        shift_step=[1,1,1,1], **kwargs)
    return model



def Caterpillar_Tx_IN1k(pretrained=False, **kwargs):
    model = Caterpillar(img_size=224, patch_size=4,
                        embed_dim=[60, 120, 240, 480],
                        depth=[2, 8, 14, 2],
                        down_stride=[4,2,2,2],
                        shift_step=[1,1,1,1], **kwargs)
    return model



def Caterpillar_T_IN1k(pretrained=False, **kwargs):
    model = Caterpillar(img_size=224, patch_size=4,
                        embed_dim=[80,160,320,640],
                        depth=[2,8,14,2],
                        down_stride=[4,2,2,2],
                        shift_step=[1,1,1,1], **kwargs)
    return model



def Caterpillar_S_IN1k(pretrained=False, **kwargs):
    model = Caterpillar(img_size=224, patch_size=4,
                        embed_dim=[96,192,384,768],
                        depth=[2,10,24,2],
                        down_stride=[4,2,2,2],
                        shift_step=[1,1,1,1], **kwargs)
    return model



def Caterpillar_B_IN1k(pretrained=False, **kwargs):
    model = Caterpillar(img_size=224, patch_size=4,
                        embed_dim=[112,224,448,896],
                        depth=[2,10,24,2],
                        down_stride=[4,2,2,2],
                        shift_step=[1,1,1,1], **kwargs)
    return model




# -------------Models for Mini_Imagenet-----------------

def Caterpillar_Mi_MIN(pretrained=False, **kwargs):
    model = Caterpillar(img_size=84, patch_size=3,
                        embed_dim=[40,80,160,320],
                        depth=[2,6,10,2],
                        down_stride=[3,2,2,1],
                        shift_step=[1,1,1,1], **kwargs)
    return model


def Caterpillar_Tx_MIN(pretrained=False, **kwargs):
    model = Caterpillar(img_size=84, patch_size=3,
                        embed_dim=[60,120,240,480],
                        depth=[2,8,14,2],
                        down_stride=[3,2,2,1],
                        shift_step=[1,1,1,1], **kwargs)
    return model


def Caterpillar_T_MIN(pretrained=False, **kwargs):
    model = Caterpillar(img_size=84, patch_size=3,
                        embed_dim=[80,160,320,640],
                        depth=[2,8,14,2],
                        down_stride=[3,2,2,1],
                        shift_step=[1,1,1,1], **kwargs)
    return model


def Caterpillar_S_MIN(pretrained=False, **kwargs):
    model = Caterpillar(img_size=84, patch_size=3,
                        embed_dim=[96,192,384,768],
                        depth=[2,10,24,2],
                        down_stride=[3,2,2,1],
                        shift_step=[1,1,1,1], **kwargs)
    return model


def Caterpillar_B_MIN(pretrained=False, **kwargs):
    model = Caterpillar(img_size=84, patch_size=3,
                        embed_dim=[112,224,448,896],
                        depth=[2,10,24,2],
                        down_stride=[3,2,2,1],
                        shift_step=[1,1,1,1], **kwargs)
    return model


# -------------Models for Cifar10-----------------

def Caterpillar_Mi_C10(pretrained=False, **kwargs):
    model = Caterpillar(img_size=32, patch_size=1,
                        embed_dim=[40,80,160,320],
                        depth=[2,6,10,2],
                        down_stride=[1,2,2,2],
                        shift_step=[1,1,1,1], **kwargs)
    return model


def Caterpillar_Tx_C10(pretrained=False, **kwargs):
    model = Caterpillar(img_size=32, patch_size=1,
                        embed_dim=[60,120,240,480],
                        depth=[2,8,14,2],
                        down_stride=[1,2,2,2],
                        shift_step=[1,1,1,1], **kwargs)
    return model


def Caterpillar_T_C10(pretrained=False, **kwargs):
    model = Caterpillar(img_size=32, patch_size=1,
                        embed_dim=[80,160,320,640],
                        depth=[2,8,14,2],
                        down_stride=[1,2,2,2],
                        shift_step=[1,1,1,1], **kwargs)
    return model


def Caterpillar_S_C10(pretrained=False, **kwargs):
    model = Caterpillar(img_size=32, patch_size=1,
                        embed_dim=[96,192,384,768],
                        depth=[2,10,24,2],
                        down_stride=[1,2,2,2],
                        shift_step=[1,1,1,1], **kwargs)
    return model

def Caterpillar_B_C10(pretrained=False, **kwargs):
    model = Caterpillar(img_size=32, patch_size=1,
                        embed_dim=[112,224,448,896],
                        depth=[2,10,24,2],
                        down_stride=[1,2,2,2],
                        shift_step=[1,1,1,1], **kwargs)
    return model


# -------------Models for Cifar100-----------------

def Caterpillar_Mi_C100(pretrained=False, **kwargs):
    model = Caterpillar(img_size=32, patch_size=1,
                        embed_dim=[40,80,160,320],
                        depth=[2,6,10,2],
                        down_stride=[1,2,2,2],
                        shift_step=[1,1,1,1], **kwargs)
    return model


def Caterpillar_Tx_C100(pretrained=False, **kwargs):
    model = Caterpillar(img_size=32, patch_size=1,
                        embed_dim=[60,120,240,480],
                        depth=[2,8,14,2],
                        down_stride=[1,2,2,2],
                        shift_step=[1,1,1,1], **kwargs)
    return model


def Caterpillar_T_C100(pretrained=False, **kwargs):
    model = Caterpillar(img_size=32, patch_size=1,
                        embed_dim=[80,160,320,640],
                        depth=[2,8,14,2],
                        down_stride=[1,2,2,2],
                        shift_step=[1,1,1,1], **kwargs)
    return model


def Caterpillar_S_C100(pretrained=False, **kwargs):
    model = Caterpillar(img_size=32, patch_size=1,
                        embed_dim=[96,192,384,768],
                        depth=[2,10,24,2],
                        down_stride=[1,2,2,2],
                        shift_step=[1,1,1,1], **kwargs)
    return model


def Caterpillar_B_C100(pretrained=False, **kwargs):
    model = Caterpillar(img_size=32, patch_size=1,
                        embed_dim=[112,224,448,896],
                        depth=[2,10,24,2],
                        down_stride=[1,2,2,2],
                        shift_step=[1,1,1,1], **kwargs)
    return model


# -------------Models for Fashion-Mnist-----------------

def Caterpillar_Mi_FM(pretrained=False, **kwargs):
    model = Caterpillar(img_size=28, patch_size=1,
                        embed_dim=[40,80,160,320],
                        depth=[2,6,10,2],
                        down_stride=[1,2,2,1],
                        shift_step=[1,1,1,1], **kwargs)
    return model


def Caterpillar_Tx_FM(pretrained=False, **kwargs):
    model = Caterpillar(img_size=28, patch_size=1,
                        embed_dim=[60,120,240,480],
                        depth=[2,8,14,2],
                        down_stride=[1,2,2,1],
                        shift_step=[1,1,1,1], **kwargs)
    return model


def Caterpillar_T_FM(pretrained=False, **kwargs):
    model = Caterpillar(img_size=28, patch_size=1,
                        embed_dim=[80,160,320,640],
                        depth=[2,8,14,2],
                        down_stride=[1,2,2,1],
                        shift_step=[1,1,1,1], **kwargs)
    return model


def Caterpillar_S_FM(pretrained=False, **kwargs):
    model = Caterpillar(img_size=28, patch_size=1,
                        embed_dim=[96,192,384,768],
                        depth=[2,10,24,2],
                        down_stride=[1,2,2,1],
                        shift_step=[1,1,1,1], **kwargs)
    return model


def Caterpillar_B_FM(pretrained=False, **kwargs):
    model = Caterpillar(img_size=28, patch_size=1,
                        embed_dim=[112,224,448,896],
                        depth=[2,10,24,2],
                        down_stride=[1,2,2,1],
                        shift_step=[1,1,1,1], **kwargs)
    return model

if __name__ == "__main__":
    device = 'cuda:0'
    # Input
    x = torch.randn([1, 1, 32, 256, 256])
    print("=" * 10, "vit_base16", "=" * 10)
    # Define the models
    model = Caterpillar3D(
            img_size=[32, 256, 256],
            patch_size=[2, 4, 4],
            embed_dim=[40, 160, 320, 768],
            depth=[2, 6, 10, 2],
            down_stride=[4, 2, 2, 2],
            shift_step=[1, 1, 1, 1]
        )
    # model = ViT(
    #     in_channels=1,
    #     img_size=(32, 256, 256),
    #     patch_size=(4, 16, 16),
    #     hidden_size=768,
    #     mlp_dim=3072,
    #     num_layers=12,
    #     num_heads=12,
    #     pos_embed="perceptron",
    #     dropout_rate=0,
    #     spatial_dims=3,
    #     classification=True,
    # )
    # NOTE: First run the model once for accurate time measurement in the following process.
    with torch.no_grad():
        macs, params = profile(model, inputs=(x, ))
        macs, params = clever_format([macs, params], "%.3f")
        print(macs)
        print(params)

