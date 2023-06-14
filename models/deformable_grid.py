import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DeformableGrid(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=1, padding=1, bias=True):

        super(DeformableGrid, self).__init__()

        self.offset_conv = nn.Conv2d(in_channels, 2, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

        nn.init.kaiming_uniform_(self.offset_conv.weight, a=math.sqrt(5))

        if self.offset_conv.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.offset_conv.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.offset_conv.bias, -bound, bound)

    def forward(self, x):
        B, C, H, W = x.shape
        
        max_offset = max(H, W)/4.
        
        offset = self.offset_conv(x).clamp(-max_offset, max_offset)

        x_offset = offset[0,0,:,:]
        y_offset = offset[0,1,:,:]
        
        xgrid, ygrid = torch.meshgrid(torch.arange(H), torch.arange(W))
        xgrid = xgrid.to(x.device)
        ygrid = ygrid.to(x.device)
        
        xgrid = xgrid + x_offset
        ygrid = ygrid + y_offset
        
        xgrid = xgrid.to(torch.long)
        xgrid[xgrid >= H] = H-1

        ygrid = ygrid.to(torch.long)
        ygrid[ygrid >= W] = W-1
        
        out = x.clone()
        out = x[:,:,xgrid,ygrid]
        
        return out

