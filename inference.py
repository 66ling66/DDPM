import torch
import matplotlib.pyplot as plt
from model import UNet
from scheduler import DDPMScheduler

class Inferencer:
    def __init__(self, checkpoint_path, device="cuda"):
        # 初始化模型和调度器
        self.model = UNet(in_channels=1).to(device)
        self.scheduler = DDPMScheduler()
        self.device = device
        
        # 加载模型权重
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"Checkpoint loss: {checkpoint['loss']:.4f}")
    
    @torch.no_grad()
    def sample(self, batch_size=1, image_size=32):
        # 从标准正态分布采样
        x = torch.randn(batch_size, 1, image_size, image_size).to(self.device)
        
        # 逐步去噪
        for t in range(self.scheduler.num_timesteps - 1, -1, -1):
            timesteps = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            predicted_noise = self.model(x, timesteps)
            x = self.scheduler.step(predicted_noise, t, x)
        
        # 将图像转换回[0, 1]范围
        x = (x + 1) / 2
        x = torch.clamp(x, 0, 1)
        return x

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 初始化推理器
    inferencer = Inferencer('checkpoint_epoch_50.pth', device)
    
    # 生成样本
    samples = inferencer.sample(batch_size=16)
    
    # 显示生成的图像
    plt.figure(figsize=(10, 10))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(samples[i, 0].cpu(), cmap="gray")
        plt.axis("off")
    plt.savefig("inference_samples.png")
    plt.close()
    print("生成的样本已保存到 inference_samples.png")

if __name__ == "__main__":
    main()