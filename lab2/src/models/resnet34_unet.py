import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        reduction_ratio=16
        mid_channels = in_channels // reduction_ratio
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, 1, bias=False)
        )

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return x * torch.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return x * torch.sigmoid(self.conv(x_cat))

class CBAM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        out = self.ca(x)
        out = self.sa(out)
        return out

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, shortcut=None):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = shortcut
    def forward(self, x):
        resdual = x if self.shortcut is None else self.shortcut(x)
        out = resdual + self.layer_stack(x)
        return F.relu(out)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super().__init__()
        mid_channels = max(in_channels // 2, out_channels)
        self.layer_stack = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size, stride),
            nn.Conv2d(in_channels, mid_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            CBAM(out_channels)
        )

    def forward(self, x):
        return self.layer_stack(x)
    

class ResNet34_UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        # ---Encoder---
        self.encode0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.encode1 = self._make_layer(64, 64, num_blocks=3, stride=1)
        self.encode2 = self._make_layer(64, 128, num_blocks=4, stride=2)
        self.encode3 = self._make_layer(128, 256, num_blocks=6, stride=2)
        self.encode4 = self._make_layer(256, 512, num_blocks=3, stride=2) 

        # ---Bottom---
        self.encode5 = self._make_layer(512, 256, num_blocks=1, stride=1)

        # ---Decoder---
        self.decode4 = DecoderBlock(768, 32, kernel_size=2, stride=2)
        self.decode3 = DecoderBlock(288, 32, kernel_size=2, stride=2)
        self.decode2 = DecoderBlock(160, 32, kernel_size=2, stride=2)
        self.decode1 = DecoderBlock(96, 32, kernel_size=2, stride=2)
        self.decode0 = DecoderBlock(32, 32, kernel_size=2, stride=2)

        # ---OutConv---
        self.final_encode = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        if stride == 2 or in_channels != out_channels:
            shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            shortcut = None

        layers = []
        layers.append(EncoderBlock(in_channels, out_channels, stride, shortcut))
        
        for _ in range(1, num_blocks):
            layers.append(EncoderBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # ---Encoder--- input (B, 3, 256, 256)
        e0 = self.encode0(x) # (B, 64, 128, 128)
        e1 = self.encode1(self.maxpool(e0)) # (B, 64, 64, 64)
        e2 = self.encode2(e1) # (B, 128, 32, 32)
        e3 = self.encode3(e2) # (B, 256, 16, 16)
        e4 = self.encode4(e3) # (B, 512, 8, 8)

        # ---Bottom---
        e5 = self.encode5(e4) # (B, 256, 8, 8)

        # ---Decoder---
        # e5(256) + e4(512) -> 768 -> upsample 32
        d4_cat = torch.cat([e5, e4], dim=1) 
        d4 = self.decode4(d4_cat) # (B, 32, 16, 16)     

        # d4(32) + e3(256) -> 288 -> upsample d3(32)
        d3_cat = torch.cat([d4, e3], dim=1)
        d3 = self.decode3(d3_cat) # (B, 32, 32, 32)

        # d3(32) + e2(128) -> 160 -> upsample d2(32)
        d2_cat = torch.cat([d3, e2], dim=1)
        d2 = self.decode2(d2_cat) # (B, 32, 64, 64)

        # d2(32) + e1(64) -> 96 -> upsample d1(32)
        d1_cat = torch.cat([d2, e1], dim=1)
        d1 = self.decode1(d1_cat) # (B, 32, 256, 256)

        # # d2(32) -> upsample d0(32)
        d0 = self.decode0(d1) # (B, 32, 256, 256)

        # ---OutConv---
        out = self.final_encode(d0)
        return self.out_conv(out) # (B, out_channels, 256, 256)
    

# test script
if __name__ == "__main__":
    dummy_input = torch.randn(2, 3, 256, 256)
    model = ResNet34_UNet()
    
    print(f"model name: {model.__class__.__name__}")
    print(f"input dim: {dummy_input.shape}")
    
    try:
        output = model(dummy_input)
        
        print(f"successfully forward pass!")
        print(f"output dim: {output.shape}")
        
        if output.shape[2:] == dummy_input.shape[2:]:
            print(f"check size: perfect")
        else:
            print(f"check size: oh no! input: {output.shape[2:]}, output: {dummy_input.shape[2:]}")

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"total parameters number: {total_params:,}")

    except Exception as e:
        print(f"something worng:")
        print(e)