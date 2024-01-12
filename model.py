import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # 编码器（收缩路径）
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 解码器（扩张路径）
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # 使用 Sigmoid 激活进行二值分割
        )

    def forward(self, x):
        x = F.interpolate(x, size=(640, 1280), mode='bilinear', align_corners=False)
        x1 = self.encoder(x)
        x2 = self.decoder(x1)
        return x2

import torch

# 创建一个 UNet 模型实例
model = UNet(in_channels=1, out_channels=1)

# 随机生成一个具有相应形状的输入张量
input_shape = (1, 1, 320, 640)  # (batch_size, channels, height, width)
input_tensor = torch.randn(input_shape)

# 使用模型进行前向传播
output_tensor = model(input_tensor)

# 打印输入和输出的形状
print("输入张量形状:", input_tensor.shape)
print("输出张量形状:", output_tensor.shape)
