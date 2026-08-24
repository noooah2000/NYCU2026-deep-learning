import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layer_stack(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        
        # ---Encoder---
        self.encode1 = DoubleConv(in_channels, 64)
        self.encode2 = DoubleConv(64, 128)
        self.encode3 = DoubleConv(128, 256)
        self.encode4 = DoubleConv(256, 512)
        self.encode5 = DoubleConv(512, 1024)
        self.downpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.5)

        # ---Decoder---
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decode4 = DoubleConv(1024, 512) 
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decode3 = DoubleConv(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decode2 = DoubleConv(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decode1 = DoubleConv(128, 64)

        # ---OutConv---
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # ---Down---
        e1 = self.encode1(x)
        p1 = self.downpool(e1)

        e2 = self.encode2(p1)
        p2 = self.downpool(e2)

        e3 = self.encode3(p2)
        p3 = self.downpool(e3)

        e4 = self.encode4(p3)
        p4 = self.downpool(e4)

        e5 = self.encode5(p4)
        e5 = self.dropout(e5)

        # ---Up---
        u4 = self.upconv4(e5)
        crop_e4 = TF.center_crop(e4, u4.shape[2:]) 
        d4 = self.decode4(torch.cat((u4, crop_e4), dim=1))

        u3 = self.upconv3(d4)
        crop_e3 = TF.center_crop(e3, u3.shape[2:])
        d3 = self.decode3(torch.cat((u3, crop_e3), dim=1))

        u2 = self.upconv2(d3)
        crop_e2 = TF.center_crop(e2, u2.shape[2:])
        d2 = self.decode2(torch.cat((u2, crop_e2), dim=1))

        u1 = self.upconv1(d2)
        crop_e1 = TF.center_crop(e1, u1.shape[2:])
        d1 = self.decode1(torch.cat((u1, crop_e1), dim=1))

        # ---OutConv---
        return self.out_conv(d1)
 
# test script
if __name__ == "__main__":
    dummy_input = torch.randn(2, 3, 256, 256)
    model = UNet()
    
    print(f"model name: {model.__class__.__name__}")
    print(f"input dim: {dummy_input.shape}")
    
    try:
        output = model(dummy_input)
        
        print(f"successfully forward pass！")
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