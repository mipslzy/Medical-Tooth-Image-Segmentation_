import torch
import torch.nn as nn

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
            DoubleConv2D(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class OutConv2D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv2D, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        # x = self.sigmoid(x)
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
        self.conv = DoubleConv2D(in_ch + skip_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class UNet2D(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(UNet2D, self).__init__()
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

# # 示例用法
# if __name__ == '__main__':
#     x = torch.randn(1, 1, 320, 640)  # 输入图像大小为320x640，通道数为1
#     net = UNet2D(in_channels=1, num_classes=1)  # 根据任务设置输出通道数和类别数
#     y = net(x)
#     print("params: ", sum(p.numel() for p in net.parameters()))
#     print(y.shape)
