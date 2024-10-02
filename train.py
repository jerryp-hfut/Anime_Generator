import torch
import torch.optim as optim
import torch.nn as nn
from generator import Generator
from discriminator import Discriminator
from data_loader import get_dataloader
import torchvision.utils as vutils
import os
import random

# 参数设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 100  # 随机噪声的维度
img_channels = 3  # 图片的通道数
img_size = 64  # 图像大小
feature_g = 64  # 生成器的特征维度
feature_d = 64  # 判别器的特征维度
batch_size = 64
epochs = 100
lr = 2e-4
beta1 = 0.5
beta2 = 0.999
sample_interval = 500  # 每隔多少步保存生成的图像

# 初始化模型
gen = Generator(latent_dim, img_channels, feature_g).to(device)
disc = Discriminator(img_channels, feature_d).to(device)

# 损失函数
criterion = torch.nn.BCEWithLogitsLoss()


# 优化器
opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(beta1, beta2))
opt_disc = optim.Adam(disc.parameters(), lr=lr, betas=(beta1, beta2))

# 加载数据
dataloader = get_dataloader(batch_size, img_size)

# 创建用于保存生成图像的文件夹
os.makedirs("generated_images", exist_ok=True)

# 固定噪声向量，用于生成对比图片
fixed_noise = torch.randn(64, latent_dim, 1, 1).to(device)

# 初始化经验回放缓存
replay_buffer = []

# 设置训练步数的动态调整
gen_update_step = 1
disc_update_step = 1

print("Training Start...")

# 训练循环
for epoch in range(epochs):
    total_loss_disc = 0
    total_loss_gen = 0
    for batch_idx, (real, _) in enumerate(dataloader):
        real = real.to(device)
        batch_size = real.size(0)

        ### 判别器训练 ###
        for _ in range(disc_update_step):
            noise = torch.randn(batch_size, latent_dim, 1, 1).to(device)
            fake = gen(noise)
            
            # 从经验回放中混入一些旧的生成样本
            if len(replay_buffer) > 0:
                replay_fake = random.sample(replay_buffer, min(batch_size // 2, len(replay_buffer)))
                replay_fake = torch.cat([tensor.to(device) for tensor in replay_fake], dim=0)  # 将 replay_fake 列表直接拼接并移动到 GPU
                fake = torch.cat([fake, replay_fake], dim=0)  # 拼接 fake 和 replay_fake
                fake = fake[:batch_size]  # 确保 batch 大小一致

            # 判别器对真实图像的损失
            disc_real = disc(real).view(-1)
            real_labels = torch.ones_like(disc_real) * torch.FloatTensor(batch_size).uniform_(0.8, 1.2).to(device)  # 加入噪声的真实标签
            loss_disc_real = criterion(disc_real, real_labels)

            # 判别器对生成图像的损失
            disc_fake = disc(fake.detach()).view(-1)
            fake_labels = torch.zeros_like(disc_fake) * torch.FloatTensor(batch_size).uniform_(0.0, 0.3).to(device)  # 加入噪声的生成标签
            loss_disc_fake = criterion(disc_fake, fake_labels)

            # 总判别器损失
            loss_disc = (loss_disc_real + loss_disc_fake) / 2

            # 优化判别器
            disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()

        ### 生成器训练 ###
        for _ in range(gen_update_step):
            noise = torch.randn(batch_size, latent_dim, 1, 1).to(device)
            fake = gen(noise)
            output = disc(fake).view(-1)

            # 生成器的损失 (使判别器认为生成的图像是真实的)
            loss_gen = criterion(output, torch.ones_like(output))

            # 优化生成器
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

        # 动态调整训练步数
        if loss_disc.item() < 0.4:
            gen_update_step = min(gen_update_step + 1, 5)  # 增加生成器训练次数
        if loss_gen.item() < 0.4:
            disc_update_step = min(disc_update_step + 1, 5)  # 增加判别器训练次数
        if loss_gen.item() > 1.0:
            gen_update_step = max(gen_update_step - 1, 1)  # 减少生成器训练次数
        if loss_disc.item() > 1.0:
            disc_update_step = max(disc_update_step - 1, 1)  # 减少判别器训练次数

        total_loss_disc += loss_disc.item()
        total_loss_gen += loss_gen.item()

        # 保存生成样本到回放缓存
        replay_buffer.append(fake.detach().cpu())
        if len(replay_buffer) > 1000:
            replay_buffer = replay_buffer[-1000:]  # 限制缓存大小

        # 定期保存生成的图片
        if batch_idx % sample_interval == 0:
            with torch.no_grad():
                fake_images = gen(fixed_noise).detach().cpu()
            vutils.save_image(fake_images, f"generated_images/sample_{epoch}_{batch_idx}.png", normalize=True)
            print(f"Saved generated image at Epoch {epoch+1}, Batch {batch_idx}")

    avg_loss_disc = total_loss_disc / len(dataloader)
    avg_loss_gen = total_loss_gen / len(dataloader)

    print(f"Epoch [{epoch+1}/{epochs}] | Avg Loss D: {avg_loss_disc:.4f} | Avg Loss G: {avg_loss_gen:.4f}")

    # 保存模型
    if (epoch + 1) % 10 == 0:  # 每10个epoch保存一次模型
        torch.save(gen.state_dict(), f"generator_epoch_{epoch+1}.pth")
        torch.save(disc.state_dict(), f"discriminator_epoch_{epoch+1}.pth")
        print("models checkpoint saved!")

# 最终保存模型
torch.save(gen.state_dict(), "generator_final.pth")
torch.save(disc.state_dict(), "discriminator_final.pth")
print("Final models saved!")
