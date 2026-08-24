import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from config import Config

# ========================================== #
# Time Embedding
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

# ========================================== #
# ResBlock & CrossAttention
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self.act = nn.SiLU()
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(self.act(self.norm1(x)))
        t_hidden = self.time_proj(self.act(t_emb))
        h = h + t_hidden.unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.shortcut(x)

class CrossAttentionBlock(nn.Module):
    def __init__(self, channels, context_dim, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads, 
            kdim=context_dim, vdim=context_dim, batch_first=True
        )

    def forward(self, x, context):
        b, c, h, w = x.shape
        x_seq = self.norm(x).view(b, c, -1).transpose(1, 2)
        attn_out, _ = self.attn(query=x_seq, key=context, value=context)
        attn_out = attn_out.transpose(1, 2).view(b, c, h, w)
        return x + attn_out

class PassThrough(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x

# ========================================== #
# DownBlock & UpBlock
class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, context_dim, use_attn=False, downsample=True):
        super().__init__()
        self.res1 = ResBlock(in_ch, out_ch, time_dim)
        self.attn1 = CrossAttentionBlock(out_ch, context_dim) if use_attn else PassThrough()
        self.res2 = ResBlock(out_ch, out_ch, time_dim)
        self.attn2 = CrossAttentionBlock(out_ch, context_dim) if use_attn else PassThrough()
        self.downsample = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1) if downsample else PassThrough()

    def forward(self, x, t, context):
        skips = []
        x = self.attn1(self.res1(x, t), context)
        skips.append(x) 
        
        x = self.attn2(self.res2(x, t), context)
        skips.append(x)
        
        x = self.downsample(x)
        return x, skips

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, time_dim, context_dim, use_attn=False, upsample=True):
        super().__init__()
        self.res1 = ResBlock(in_ch + skip_ch, out_ch, time_dim)
        self.attn1 = CrossAttentionBlock(out_ch, context_dim) if use_attn else PassThrough()
        self.res2 = ResBlock(out_ch + skip_ch, out_ch, time_dim)
        self.attn2 = CrossAttentionBlock(out_ch, context_dim) if use_attn else PassThrough()
        
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest") if upsample else PassThrough()
        self.conv_up = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1) if upsample else PassThrough()

    def forward(self, x, skips, t, context):
        x = torch.cat([x, skips.pop()], dim=1)
        x = self.attn1(self.res1(x, t), context)
        
        x = torch.cat([x, skips.pop()], dim=1)
        x = self.attn2(self.res2(x, t), context)
        
        x = self.conv_up(self.upsample(x))
        return x

# ========================================== #
# UNet
class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, config=Config):
        super().__init__()
        time_dim = config.TIME_DIM
        context_dim = config.CONTEXT_DIM
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim // 2),
            nn.Linear(time_dim // 2, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        
        self.cond_proj = nn.Linear(config.COND_DIM, context_dim)
        self.init_conv = nn.Conv2d(c_in, 128, kernel_size=3, padding=1)
        
        # Down Block
        self.down1 = DownBlock(128, 128, time_dim, context_dim, use_attn=False, downsample=True)
        self.down2 = DownBlock(128, 256, time_dim, context_dim, use_attn=True,  downsample=True)
        self.down3 = DownBlock(256, 512, time_dim, context_dim, use_attn=True,  downsample=True)
        self.down4 = DownBlock(512, 512, time_dim, context_dim, use_attn=False, downsample=False)
        
        # Middle Block
        self.mid_res1 = ResBlock(512, 512, time_dim)
        self.mid_attn = CrossAttentionBlock(512, context_dim)
        self.mid_res2 = ResBlock(512, 512, time_dim)
        
        # Up Block
        self.up4 = UpBlock(512, 512, 512, time_dim, context_dim, use_attn=False, upsample=True)
        self.up3 = UpBlock(512, 512, 256, time_dim, context_dim, use_attn=True,  upsample=True)
        self.up2 = UpBlock(256, 256, 128, time_dim, context_dim, use_attn=True,  upsample=True)
        self.up1 = UpBlock(128, 128, 128, time_dim, context_dim, use_attn=False, upsample=False)
        
        self.final_norm = nn.GroupNorm(32, 128)
        self.final_conv = nn.Conv2d(128, c_out, kernel_size=3, padding=1)

    def forward(self, x, time, cond):
        t = self.time_mlp(time)
        context = self.cond_proj(cond).unsqueeze(1) 
        x = self.init_conv(x)
        
        x, s1 = self.down1(x, t, context)
        x, s2 = self.down2(x, t, context)
        x, s3 = self.down3(x, t, context)
        x, s4 = self.down4(x, t, context)
        
        x = self.mid_res1(x, t)
        x = self.mid_attn(x, context)
        x = self.mid_res2(x, t)
        
        x = self.up4(x, s4, t, context)
        x = self.up3(x, s3, t, context)
        x = self.up2(x, s2, t, context)
        x = self.up1(x, s1, t, context)
        
        x = F.silu(self.final_norm(x))
        return self.final_conv(x)