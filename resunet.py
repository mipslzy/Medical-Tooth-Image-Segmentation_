import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        # Additional convolution to match the number of channels
        if in_channels != out_channels:
            self.match_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        # Apply additional convolution if necessary
        if residual.shape[1] != out.shape[1]:
            residual = self.match_conv(residual)

        out += residual
        out = self.relu(out)
        return out


class ResUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResUNet, self).__init__()

        # Encoder
        self.enc1 = ResidualBlock(in_channels, 64)
        self.enc1_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = ResidualBlock(64, 128)
        self.enc2_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = ResidualBlock(128, 256)

        # Decoder
        self.dec2_upsample = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(256, 128)

        self.dec1_upsample = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(128, 64)

        # Output layer
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc1_pool = self.enc1_pool(enc1)

        enc2 = self.enc2(enc1_pool)
        enc2_pool = self.enc2_pool(enc2)

        bottleneck = self.bottleneck(enc2_pool)

        dec2_upsample = self.dec2_upsample(bottleneck)
        dec2 = self.dec2(torch.cat([enc2, dec2_upsample], dim=1))

        dec1_upsample = self.dec1_upsample(dec2)
        dec1 = self.dec1(torch.cat([enc1, dec1_upsample], dim=1))

        output = self.out_conv(dec1)
        return output

# # Instantiate the model
# model = ResUNet(in_channels=1, out_channels=1)
#
# # Print the model architecture
# print(model)
