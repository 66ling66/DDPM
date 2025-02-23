import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import UNet
from scheduler import DDPMScheduler

class Trainer:
    def __init__(self, model, scheduler, device="cuda"):
        self.model = model.to(device)
        self.scheduler = scheduler
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def train_step(self, batch):
        # 将图像移动到设备并归一化到[-1, 1]
        x_0 = batch[0].to(self.device)
        x_0 = x_0 * 2 - 1
        batch_size = x_0.shape[0]

        # 采样时间步
        t = torch.randint(0, self.scheduler.num_timesteps, (batch_size,), device=self.device).long()

        # 生成噪声
        noise = torch.randn_like(x_0)

        # 添加噪声
        x_t = self.scheduler.add_noise(x_0, noise, t)

        # 预测噪声
        predicted_noise = self.model(x_t, t)

        # 计算损失
        loss = F.mse_loss(predicted_noise, noise)

        # 优化器步骤
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, train_loader, num_epochs):
        for epoch in range(num_epochs):
            pbar = tqdm(train_loader)
            epoch_loss = 0
            for batch in pbar:
                loss = self.train_step(batch)
                epoch_loss += loss
                pbar.set_description(f"Epoch {epoch}, Loss: {loss:.4f}")
            avg_loss = epoch_loss/len(train_loader)
            print(f"Epoch {epoch} average loss: {avg_loss:.4f}")
            
            # 每10个epoch保存一次权重
            if (epoch + 1) % 10 == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': avg_loss,
                }
                torch.save(checkpoint, f'checkpoint_epoch_{epoch+1}.pth')
                print(f"Saved checkpoint at epoch {epoch+1}")

    @torch.no_grad()
    def sample(self, batch_size=1, image_size=32):
        # 从标准正态分布采样
        x = torch.randn(batch_size, 1, image_size, image_size).to(self.device)

        # 逐步去噪
        for t in tqdm(range(self.scheduler.num_timesteps - 1, -1, -1)):
            timesteps = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            predicted_noise = self.model(x, timesteps)
            x = self.scheduler.step(predicted_noise, t, x)

        # 将图像转换回[0, 1]范围
        x = (x + 1) / 2
        x = torch.clamp(x, 0, 1)
        return x

# def test_trainer():
#     # 设置设备
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"使用设备: {device}")

#     # 创建小型测试数据集
#     batch_size = 4
#     image_size = 32
#     test_images = torch.randn(batch_size, 1, image_size, image_size).to(device)  # [4, 1, 28, 28]
#     test_labels = torch.zeros(batch_size).to(device)
#     batch = (test_images, test_labels)  # 直接构造batch数据

#     # 初始化模型和调度器
#     model = UNet(in_channels=1)
#     scheduler = DDPMScheduler()
#     trainer = Trainer(model, scheduler, device)

#     # 测试train_step
#     print("\n测试train_step:")
#     loss = trainer.train_step(batch)
#     print(f"训练步骤损失: {loss}")

#     # 验证模型和数据是否在正确的设备上
#     print("\n验证设备使用情况:")
#     print(f"模型设备: {next(trainer.model.parameters()).device}")
#     print(f"优化器状态: {trainer.optimizer.state_dict()['param_groups'][0]['lr']}")

#     # 测试采样功能
#     print("\n测试采样功能:")
#     samples = trainer.sample(batch_size=2, image_size=32)
#     print(f"生成样本形状: {samples.shape}")
#     print(f"样本值范围: [{samples.min():.4f}, {samples.max():.4f}]")

#     print("\n所有测试完成!")

# if __name__ == "__main__":
#     test_trainer()

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载MNIST数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Pad(2),  # 在每个边缘添加2个像素的padding，将28x28变为32x32
    ])
    dataset = datasets.MNIST(root="data", train=True, transform=transform, download=True)
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)

    # 初始化模型和调度器
    model = UNet(in_channels=1)
    scheduler = DDPMScheduler()

    # 初始化训练器
    trainer = Trainer(model, scheduler, device)

    # 训练模型
    trainer.train(train_loader, num_epochs=50)

    # 生成样本
    samples = trainer.sample(batch_size=16)

    # 显示生成的图像
    plt.figure(figsize=(10, 10))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(samples[i, 0].cpu(), cmap="gray")
        plt.axis("off")
    plt.savefig("samples.png")
    plt.close()

if __name__ == "__main__":
    main()