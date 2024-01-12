import torch
import torch.nn as nn

class InitialBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InitialBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.prelu(x)
        return x

class RegularBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super(RegularBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.prelu1 = nn.PReLU(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
        self.prelu2 = nn.PReLU(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(out_channels)
        self.prelu3 = nn.PReLU(out_channels)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.prelu1(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.prelu2(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)

        x += residual
        x = self.prelu3(x)

        return x

class DownsamplingBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsamplingBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.prelu1 = nn.PReLU(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
        self.prelu2 = nn.PReLU(out_channels)

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        residual = self.residual(x)

        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.prelu1(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.prelu2(x)

        x += residual

        return x

class UNetENet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(UNetENet, self).__init__()

        # Initial Block
        self.initial = InitialBlock(in_channels, 16)

        # Stage 1
        self.stage1_0 = DownsamplingBottleneck(16, 64)
        self.stage1_1 = RegularBottleneck(64, 64, dilation=1)
        self.stage1_2 = RegularBottleneck(64, 64, dilation=1)
        self.stage1_3 = RegularBottleneck(64, 64, dilation=1)
        self.stage1_4 = RegularBottleneck(64, 64, dilation=1)

        # Stage 2
        self.stage2_0 = DownsamplingBottleneck(64, 128)
        self.stage2_1 = RegularBottleneck(128, 128, dilation=2)
        self.stage2_2 = RegularBottleneck(128, 128, dilation=2)
        self.stage2_3 = RegularBottleneck(128, 128, dilation=2)
        self.stage2_4 = RegularBottleneck(128, 128, dilation=2)
        self.stage2_5 = RegularBottleneck(128, 128, dilation=2)

        # Stage 3
        self.stage3_0 = DownsamplingBottleneck(128, 128)
        self.stage3_1 = RegularBottleneck(128, 128, dilation=4)
        self.stage3_2 = RegularBottleneck(128, 128, dilation=4)
        self.stage3_3 = RegularBottleneck(128, 128, dilation=4)
        self.stage3_4 = RegularBottleneck(128, 128, dilation=4)
        self.stage3_5 = RegularBottleneck(128, 128, dilation=4)
        self.stage3_6 = RegularBottleneck(128, 128, dilation=4)
        self.stage3_7 = RegularBottleneck(128, 128, dilation=4)

        # Stage 4
        self.stage4_0 = DownsamplingBottleneck(128, 128)
        self.stage4_1 = RegularBottleneck(128, 128, dilation=8)
        self.stage4_2 = RegularBottleneck(128, 128, dilation=8)

        # Upsampling
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.upconv2 = nn.ConvTranspose2d(64, 16, kernel_size=4, stride=2, padding=1)

        # Final Convolution
        self.final_conv = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        # Initial Block
        x = self.initial(x)

        # Stage 1
        stage1_0 = self.stage1_0(x)
        stage1_1 = self.stage1_1(stage1_0)
        stage1_2 = self.stage1_2(stage1_1)
        stage1_3 = self.stage1_3(stage1_2)
        stage1_4 = self.stage1_4(stage1_3)

        # Stage 2
        stage2_0 = self.stage2_0(stage1_4)
        stage2_1 = self.stage2_1(stage2_0)
        stage2_2 = self.stage2_2(stage2_1)
        stage2_3 = self.stage2_3(stage2_2)
        stage2_4 = self.stage2_4(stage2_3)
        stage2_5 = self.stage2_5(stage2_4)

        # Stage 3
        stage3_0 = self.stage3_0(stage2_5)
        stage3_1 = self.stage3_1(stage3_0)
        stage3_2 = self.stage3_2(stage3_1)
        stage3_3 = self.stage3_3(stage3_2)
        stage3_4 = self.stage3_4(stage3_3)
        stage3_5 = self.stage3_5(stage3_4)
        stage3_6 = self.stage3_6(stage3_5)
        stage3_7 = self.stage3_7(stage3_6)

        # Stage 4
        stage4_0 = self.stage4_0(stage3_7)
        stage4_1 = self.stage4_1(stage4_0)
        stage4_2 = self.stage4_2(stage4_1)

        # Upsampling
        upconv1 = self.upconv1(stage4_2, output_size=stage3_7.size())
        upconv2 = self.upconv2(upconv1, output_size=x.size())

        # Final Convolution
        final_conv = self.final_conv(upconv2)

        return final_conv

# 示例用法
if __name__ == '__main__':
    x = torch.randn(1, 1, 320, 640)  # 输入图像大小为320x640，通道数为1
    model = UNetENet(in_channels=1, num_classes=1)  # 根据任务设置输出通道数和类别数
    y = model(x)
    print("Output shape:", y.shape)