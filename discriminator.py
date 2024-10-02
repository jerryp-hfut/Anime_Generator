import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, img_channels, feature_d):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            # 输入: img_channels x 64 x 64
            nn.Conv2d(img_channels, feature_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # (feature_d) x 32 x 32
            nn.Conv2d(feature_d, feature_d * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_d * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (feature_d * 2) x 16 x 16
            nn.Conv2d(feature_d * 2, feature_d * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_d * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (feature_d * 4) x 8 x 8
            nn.Conv2d(feature_d * 4, feature_d * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_d * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # (feature_d * 8) x 4 x 4
            nn.Conv2d(feature_d * 8, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
            # 输出: 1 x 1 x 1 (是否为真实图像的概率)
        )

    def forward(self, x):
        return self.net(x).view(-1, 1)
