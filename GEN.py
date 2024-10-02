import torch
from generator import Generator
from torchvision.utils import save_image

# 参数
latent_dim = 100
img_channels = 3
feature_g = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载生成器
gen = Generator(latent_dim, img_channels, feature_g).to(device)
gen.load_state_dict(torch.load("generator.pth"))
gen.eval()

# 随机生成图片
with torch.no_grad():
    noise = torch.randn(64, latent_dim, 1, 1).to(device)
    fake_images = gen(noise)
    save_image(fake_images, "generated_images.png", nrow=8, normalize=True)
