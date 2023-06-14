import torch
from torch import nn, einsum
import torch.nn.functional as F
from functools import partial
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
from models.deformable_grid import DeformableGrid
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# attention related classes

class Attention(nn.Module):
    def __init__(self, dim, dim_head = 32, dropout = 0., window_size = 7):
        super().__init__()
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'

        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)

        self.attend = nn.Sequential(
            nn.Softmax(dim = -1),
            nn.Dropout(dropout)
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias = False),
            nn.Dropout(dropout)
        )

        # relative positional bias

        self.rel_pos_bias = nn.Embedding((2 * window_size - 1) ** 2, self.heads)

        pos = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(pos, pos, indexing = 'ij'))
        grid = rearrange(grid, 'c i j -> (i j) c')
        rel_pos = rearrange(grid, 'i ... -> i 1 ...') - rearrange(grid, 'j ... -> 1 j ...')
        rel_pos += window_size - 1
        rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim = -1)

        self.register_buffer('rel_pos_indices', rel_pos_indices, persistent = False)

    def forward(self, x):
        batch, height, width, window_height, window_width, _, device, h = *x.shape, x.device, self.heads

        # flatten
        x = rearrange(x, 'b x y w1 w2 d -> (b x y) (w1 w2) d')

        # project for queries, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        # split heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d ) -> b h n d', h = h), (q, k, v))

        # scale
        q = q * self.scale

        # sim
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # add positional bias
        bias = self.rel_pos_bias(self.rel_pos_indices)
        sim = sim + rearrange(bias, 'i j h -> h i j')

        # attention
        attn = self.attend(sim)

        # aggregate
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads
        out = rearrange(out, 'b h (w1 w2) d -> b w1 w2 (h d)', w1 = window_height, w2 = window_width)

        # combine heads out
        out = self.to_out(out)

        return rearrange(out, '(b x y) ... -> b x y ...', x = height, y = width)


class Conv_FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):       
        return self.net(x)


class EncoderBlock(nn.Module):
    def __init__(self, dim, dim_head=8, grid_dropout=0., window_size=4, drop_path=0.1):
        super().__init__()
        self.window_size = window_size
        layer_scale_init_value = 1e-6

        self.pos = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, padding_mode='replicate', bias=False, groups=dim)

        self.layer_norm1 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.layer_norm2 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.layer_norm0 = LayerNorm(dim, eps=1e-6, data_format="channels_first")

        self.mlp = Conv_FeedForward(dim=dim)        

        # sparse attention
        self.deform_grid = DeformableGrid(dim)
        self.attn = nn.Sequential(
                    Rearrange('b d (w1 x) (w2 y) -> b x y w1 w2 d', w1 = self.window_size, w2 = self.window_size),
                    Attention(dim = dim, dim_head = dim_head, dropout = grid_dropout, window_size = self.window_size),
                    Rearrange('b x y w1 w2 d -> b d (w1 x) (w2 y)'),
                )
        self.act = nn.GELU()

        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.drop_path_1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.drop_path_2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x, H, W):
        B, C, H, W = x.shape

        skip = x
        x = self.layer_norm0(skip)
        x = skip + self.act(self.pos(x))
        
        skip = x       
        x = self.layer_norm1(skip)
        x = self.deform_grid(x)
        x = self.attn(x)
        x = self.drop_path_1(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * x)
        out = x  + skip

        x = self.layer_norm2(out)
        x = self.mlp(x)
        x = self.drop_path_2(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * x)
        out = out + x

        return out


class CEFF(nn.Module):
    def __init__(self, in_channels, height=2, reduction=8, bias=False):
        super(CEFF, self).__init__()
        
        self.height = height
        d = max(int(in_channels/reduction),4)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.LeakyReLU(0.2))

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1,bias=bias))
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats =  inp_feats[0].shape[1]

        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])
        
        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        
        attention_vectors = self.softmax(attention_vectors)
        
        feats_V = torch.sum(inp_feats*attention_vectors, dim=1)
        
        return feats_V        


class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


