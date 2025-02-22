import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class SinusoidalPositionEmbeddings(nn.Module):
    """时间步长的正弦位置嵌入"""
    def __init__(self, dim):
        """
        初始化函数
        Args:
            dim: 嵌入维度
        """
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        前向传播函数
        Args:
            time: 时间步长张量
        Returns:
            embeddings: 时间步长的正弦位置嵌入
        """
        # 获取设备信息
        device = time.device
        # 计算半维度(因为要分别计算sin和cos)
        half_dim = self.dim // 2
        # 计算嵌入的基础值
        embeddings = math.log(10000) / (half_dim - 1)
        # 生成不同频率的正弦波
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        # 将时间信息与频率信息结合
        embeddings = time[:, None] * embeddings[None, :]
        # 连接sin和cos得到最终的位置嵌入
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        
    def forward(self, x, t):
        h = self.bnorm1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(..., ) + (None, ) * 2]
        h = h + time_emb
        h = self.bnorm2(self.relu(self.conv2(h)))
        return self.transform(h)

class UNet(nn.Module):
    def __init__(self, in_channels=1, time_dim=256):
        super().__init__()
        
        # Time embedding
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )
        
        # Initial projection
        self.conv0 = nn.Conv2d(in_channels, 64, 3, padding=1)

        # Encoder
        self.downs = nn.ModuleList([
            Block(64, 128, time_dim),
            Block(128, 256, time_dim),
            Block(256, 512, time_dim),
            Block(512, 1024, time_dim),
        ])
        
        # Bottleneck
        self.bottleneck1 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.bottleneck2 = nn.Conv2d(1024, 1024, 3, padding=1)
        
        # Decoder
        self.ups = nn.ModuleList([
            Block(1024, 512, time_dim, up=True),
            Block(512, 256, time_dim, up=True),
            Block(256, 128, time_dim, up=True),
            Block(128, 64, time_dim, up=True),
        ])

        # Final projection
        self.output = nn.Conv2d(64, in_channels, 1)

    def forward(self, x, timestep):
        # Embed time
        t = self.time_mlp(timestep)
        
        # Initial conv
        x = self.conv0(x)
        
        # Encoder
        residuals = []
        for down in self.downs:
            x = down(x, t)
            residuals.append(x)
        
        # Bottleneck
        x = F.relu(self.bottleneck1(x))
        x = F.relu(self.bottleneck2(x))
        
        # Decoder
        for up, residual in zip(self.ups, residuals[::-1]):
            x = torch.cat((x, residual), dim=1)
            x = up(x, t)
            
        return self.output(x)

if __name__ == '__main__':
    # 测试SinusoidalPositionEmbeddings类
    import matplotlib.pyplot as plt
    import numpy as np

    # 创建实例
    dim = 32
    embedder = SinusoidalPositionEmbeddings(dim)

    # 测试单个时间步
    t = torch.tensor([0])
    output = embedder(t)
    print(f"单个时间步输出维度: {output.shape}")
    assert output.shape == (1, dim), f"期望维度(1, {dim})，但得到{output.shape}"

    # 测试批量时间步
    batch_size = 16
    t = torch.arange(batch_size)
    output = embedder(t)
    print(f"批量时间步输出维度: {output.shape}")
    assert output.shape == (batch_size, dim), f"期望维度({batch_size}, {dim})，但得到{output.shape}"

    # 可视化嵌入结果
    t = torch.arange(100)
    embeddings = embedder(t).detach().numpy()

    plt.figure(figsize=(10, 4))
    plt.imshow(embeddings, aspect='auto', cmap='viridis')
    plt.colorbar(label='Embedding Value')
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Timestep')
    plt.title('Sinusoidal Position Embeddings Visualization')
    plt.savefig('position_embeddings.png')
    plt.close()

    print("所有测试通过！")