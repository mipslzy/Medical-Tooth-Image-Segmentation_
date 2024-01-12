# FCNFCN（Fully Convolutional Networks）

import torch
import torch.nn as nn
import torch.nn.functional as F

class FCN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(FCN, self).__init__()

        # 编码器部分
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 解码器部分
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = F.interpolate(x, size=(640, 1280), mode='bilinear', align_corners=False)
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return x

# # 模型输入和输出的通道数和类别数
# in_channels = 1  # 例如，RGB图像
# num_classes = 1  # 例如，用于分割的类别数
#
# # 创建FCN模型
# fcn_model = FCN(in_channels, num_classes)
# # 随机生成一个具有相应形状的输入张量
# input_shape = (1, 1, 320, 640)  # (batch_size, channels, height, width)
# input_tensor = torch.randn(input_shape)
#
# # 使用模型进行前向传播
# output_tensor = fcn_model(input_tensor)
#
# # 打印输入和输出的形状
# print("输入张量形状:", input_tensor.shape)
# print("输出张量形状:", output_tensor.shape)

