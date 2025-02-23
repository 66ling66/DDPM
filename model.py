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
    """UNet的基本构建块，包含时间嵌入和上/下采样功能
    该模块可以根据配置执行上采样或下采样操作，并将时间信息注入到特征图中
    """
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        """
        Args:
            in_ch (int): 输入通道数
            out_ch (int): 输出通道数
            time_emb_dim (int): 时间嵌入的维度
            up (bool): 是否执行上采样，False则执行下采样
        """
        super().__init__()
        # 时间嵌入的线性投影层
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        
        # 根据up参数选择上采样或下采样配置
        if up:
            # 上采样时需要考虑残差连接带来的通道数
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            # 使用转置卷积进行上采样，确保输出尺寸与残差连接匹配
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            # 下采样时使用普通卷积
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            # 使用步长为2的卷积进行下采样，padding=1以保持特征图尺寸的一致性
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
            
        # 第二个卷积层保持通道数不变
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        # 批归一化层
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        # 激活函数
        self.relu = nn.ReLU()
        
    def forward(self, x, t):
        """前向传播函数
        Args:
            x (Tensor): 输入特征图 [B, C, H, W]
            t (Tensor): 时间嵌入 [B, time_emb_dim]
        Returns:
            Tensor: 经过处理的特征图，尺寸在H和W上变为原来的1/2（下采样）或2倍（上采样）
        """
        # 第一个卷积块：卷积+激活+归一化
        h = self.bnorm1(self.relu(self.conv1(x)))
        # 处理时间嵌入
        time_emb = self.relu(self.time_mlp(t))
        # 扩展时间嵌入的维度以匹配特征图 [B, C] -> [B, C, 1, 1]
        time_emb = time_emb.unsqueeze(-1).unsqueeze(-1)
        # 将时间信息注入特征图
        h = h + time_emb
        # 第二个卷积块
        h = self.bnorm2(self.relu(self.conv2(h)))
        # 上采样或下采样
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
        residuals = residuals[::-1]
        for i, (up, residual) in enumerate(zip(self.ups, residuals)):
            
            # print(f"x_shape:{x.shape} residual_shape:{residual.shape}")
            x = up(torch.cat([x, residual], dim=1), t)
            
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
    print(f"输入维度：{t.shape} 单个时间步输出维度: {output.shape}")
    assert output.shape == (1, dim), f"期望维度(1, {dim})，但得到{output.shape}"

    # 测试批量时间步
    batch_size = 16
    t = torch.arange(batch_size)
    output = embedder(t)
    print(f"输入维度：{t.shape} 批量时间步输出维度: {output.shape}")
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

    print("\n测试Block类")
    # 测试下采样Block
    print("\n测试下采样Block:")
    batch_size = 4
    in_channels = 64
    out_channels = 128
    time_dim = 256
    h, w = 32, 32

    # 创建输入张量
    x = torch.randn(batch_size, in_channels, h, w)
    t = torch.randn(batch_size, time_dim)

    # 创建下采样Block实例
    down_block = Block(in_channels, out_channels, time_dim, up=False)
    print(f"输入特征图维度: {x.shape}")
    print(f"输入时间嵌入维度: {t.shape}")

    # 前向传播
    output = down_block(x, t)
    print(f"下采样输出维度: {output.shape}")

    # 测试上采样Block
    print("\n测试上采样Block:")
    in_channels = 128
    out_channels = 64
    h, w = 16, 16  # 输入较小的特征图

    # 创建输入张量（注意上采样时要考虑skip connection，所以输入通道要翻倍）
    x = torch.randn(batch_size, in_channels*2, h, w)  # 通道数翻倍
    t = torch.randn(batch_size, time_dim)

    # 创建上采样Block实例
    up_block = Block(in_channels, out_channels, time_dim, up=True)
    print(f"输入特征图维度: {x.shape}")
    print(f"输入时间嵌入维度: {t.shape}")

    # 前向传播
    output = up_block(x, t)
    print(f"上采样输出维度: {output.shape}")

    print("\n测试UNet类")
    # 创建UNet实例
    batch_size = 4
    in_channels = 1
    time_dim = 256
    h, w = 32, 32

    # 创建模型实例
    model = UNet(in_channels=in_channels, time_dim=time_dim)
    # print(f"\n模型结构:\n{model}")

    # 创建输入张量
    x = torch.randn(batch_size, in_channels, h, w)
    t = torch.randint(0, 1000, (batch_size,))
    print(f"\n输入图像维度: {x.shape}")
    print(f"输入时间步长维度: {t.shape}")

    # 前向传播
    output = model(x, t)
    print(f"输出维度: {output.shape}")
    assert output.shape == x.shape, f"输出维度{output.shape}与输入维度{x.shape}不匹配"

    # 测试不同尺寸的输入
    test_sizes = [(1, 1, 64, 64), (8, 1, 16, 16), (2, 1, 128, 128)]
    print("\n测试不同输入尺寸:")
    for size in test_sizes:
        x = torch.randn(*size)
        t = torch.randint(0, 1000, (size[0],))
        output = model(x, t)
        print(f"输入维度: {x.shape} -> 输出维度: {output.shape}")
        assert output.shape == x.shape, f"输出维度{output.shape}与输入维度{x.shape}不匹配"

    print("\n所有测试通过！")