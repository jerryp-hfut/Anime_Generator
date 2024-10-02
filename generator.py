import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim, img_channels, feature_g):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # 输入：latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, feature_g * 16, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(feature_g * 16),
            nn.ReLU(True),
            # (feature_g * 16) x 4 x 4
            nn.ConvTranspose2d(feature_g * 16, feature_g * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_g * 8),
            nn.ReLU(True),
            # (feature_g * 8) x 8 x 8
            nn.ConvTranspose2d(feature_g * 8, feature_g * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_g * 4),
            nn.ReLU(True),
            # (feature_g * 4) x 16 x 16
            nn.ConvTranspose2d(feature_g * 4, feature_g * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_g * 2),
            nn.ReLU(True),
            # (feature_g * 2) x 32 x 32
            nn.ConvTranspose2d(feature_g * 2, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
            # img_channels x 64 x 64
        )

    def forward(self, x):
        return self.net(x)
