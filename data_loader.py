import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

# 自定义图像加载函数
def load_image(path):
    return Image.open(path).convert("RGB")

def get_dataloader(batch_size, img_size):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # 使用 DatasetFolder 直接加载没有分类的图像
    dataset = datasets.DatasetFolder(root="archive", loader=load_image, extensions=("png", "jpg", "jpeg"), transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
