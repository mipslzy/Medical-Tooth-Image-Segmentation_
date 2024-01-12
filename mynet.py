import torch
import torch.nn as nn

class FeatureActivationModule2D(nn.Module):
    def __init__(self, in_channels):
        super(FeatureActivationModule2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.conv(x))

class HDCModule2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HDCModule2D, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3x3_d1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv3x3_d2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2)
        self.conv3x3_d4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=4, dilation=4)

    def forward(self, x):
        x1 = self.conv1x1(x)
        x2 = self.conv3x3_d1(x)
        x3 = self.conv3x3_d2(x)
        x4 = self.conv3x3_d4(x)
        O = torch.cat([x1, x2, x3, x4], dim=1)
        return torch.cat([x1, x2, x3, x4], dim=1)

class CBAMModule2D(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CBAMModule2D, self).__init__()
        self.channel_attention = ChannelAttention2D(in_channels, reduction)
        self.spatial_attention = SpatialAttention2D()

    def forward(self, x):
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        return x

class ChannelAttention2D(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention2D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)
        avg_out = self.fc2(self.relu1(self.fc1(avg_pool)))
        max_out = self.fc2(self.relu1(self.fc1(max_pool)))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention2D(nn.Module):
    def __init__(self):
        super(SpatialAttention2D, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class InConv2D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InConv2D, self).__init__()
        self.conv = DoubleConv2D(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x

class Down2D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down2D, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2, 2),
            FeatureActivationModule2D(in_ch),
            DoubleConv2D(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class OutConv2D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv2D, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class DoubleConv2D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv2D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class Up2D(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super(Up2D, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            HDCModule2D(in_ch + skip_ch, skip_ch // 2),
            CBAMModule2D(skip_ch * 2),
            DoubleConv2D(in_ch + skip_ch, out_ch)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class myUNet2D(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(myUNet2D, self).__init__()
        features = [32, 64, 128, 256]

        self.inc = InConv2D(in_channels, features[0])
        self.down1 = Down2D(features[0], features[1])
        self.down2 = Down2D(features[1], features[2])
        self.down3 = Down2D(features[2], features[3])
        self.down4 = Down2D(features[3], features[3])

        self.up1 = Up2D(features[3], features[3], features[2])
        self.up2 = Up2D(features[2], features[2], features[1])
        self.up3 = Up2D(features[1], features[1], features[0])
        self.up4 = Up2D(features[0], features[0], features[0])
        self.outc = OutConv2D(features[0], num_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

# # Example usage
# if __name__ == '__main__':
#     x = torch.randn(1, 1, 320, 640)
#     net = myUNet2D(in_channels=1, num_classes=4)  # Assuming you want 4 output channels for segmentation
#     y = net(x)
#     print("params: ", sum(p.numel() for p in net.parameters()))
#     print(y.shape)
