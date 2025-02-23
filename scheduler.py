import torch
import numpy as np

import torch

class DDPMScheduler:
    """DDPM (Denoising Diffusion Probabilistic Models) 调度器
    
    该类实现了DDPM中的噪声调度机制，包括：
      1. 前向扩散过程：逐步将图像加噪至纯噪声
      2. 反向扩散过程：通过预测噪声逐步将噪声图像恢复为原图
      
    主要参数：
      num_timesteps: 扩散步数，决定加噪和去噪的精细程度
      beta_start: β调度的起始值
      beta_end: β调度的终止值
    """
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.num_timesteps = num_timesteps
        
        # β值线性从 beta_start 增长到 beta_end
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        # α = 1 - β
        self.alphas = 1 - self.betas
        # 累积乘积：\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        # 前一时刻的累积α：\bar{\alpha}_{t-1}
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.]), self.alphas_cumprod[:-1]])
        
        # 预计算各种系数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1 / self.alphas)
        
        # 后验方差：\sigma_t^2 = \beta_t \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
    
    def add_noise(self, x_0, noise, timesteps):
        """前向扩散过程，将图像逐步加噪
        
        参数：
          x_0: 原始图像
          noise: 要添加的噪声（与 x_0 同形状）
          timesteps: 当前时间步（整数或整数张量）
          
        返回：
          x_t: 加噪后的图像
        """
        # 确保张量在正确的设备上
        device = x_0.device
        sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        
        # 调整当前时刻的系数维度
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[timesteps].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[timesteps].reshape(-1, 1, 1, 1)
        
        # 计算 x_t = √{\bar{α}_t} x_0 + √{1- \bar{α}_t} noise
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t
    
    def step(self, model_output, timestep, sample):
        """反向扩散过程中的单步去噪
        
        参数：
          model_output: 模型预测的噪声 \(\epsilon_\theta(x_t,t)\)
          timestep: 当前时间步 t（标量整数）
          sample: 当前时刻的图像 \(x_t\)（带噪）
          
        返回：
          pred_prev_sample: 预测的上一时刻图像 \(x_{t-1}\)
        """
        t = timestep
        # 当前时刻的系数
        alpha_t = self.alphas[t]                     # \(\alpha_t\)
        alpha_t_cumprod = self.alphas_cumprod[t]       # \(\bar{\alpha}_t\)
        beta_t = self.betas[t]                         # \(\beta_t\)
        # 前一时刻的累积 α
        alpha_t_cumprod_prev = self.alphas_cumprod_prev[t]
        
        # 1. 预测原始图像 \(\hat{x}_0\)：
        #    \(\hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\,\epsilon_\theta(x_t,t)}{\sqrt{\bar{\alpha}_t}}\)
        pred_x0 = (sample - self.sqrt_one_minus_alphas_cumprod[t] * model_output) / self.sqrt_alphas_cumprod[t]
        
        # 2. 根据原文计算后验均值：
        #    \(\mu_\theta(x_t,t) = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}\,\hat{x}_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\,x_t\)
        pred_mean = (torch.sqrt(alpha_t_cumprod_prev) * beta_t * pred_x0 +
                     torch.sqrt(alpha_t) * (1 - alpha_t_cumprod_prev) * sample) / (1 - alpha_t_cumprod)
        
        # 3. 添加噪声得到 x_{t-1}：
        # 当 t>0 时，采样噪声 z ~ N(0,I)，否则 z = 0
        if t > 0:
            noise = torch.randn_like(model_output)
        else:
            noise = 0.0
        
        # 反向更新：
        # \(x_{t-1} = \mu_\theta(x_t,t) + \sigma_t\, z\)，其中 \(\sigma_t = \sqrt{\mathrm{posterior\_variance}_t}\)
        variance = torch.tensor(0.0).to(sample.device) if t == 0 else self.posterior_variance[t].to(sample.device)
        std = torch.sqrt(variance)
        pred_prev_sample = pred_mean + std * noise
        
        return pred_prev_sample

if __name__ == '__main__':
    # 创建一个简单的测试图像
    batch_size = 2
    image_size = 4
    channels = 3
    test_image = torch.randn(batch_size, channels, image_size, image_size)
    print(f"测试图像形状: {test_image.shape}")
    
    # 初始化调度器
    scheduler = DDPMScheduler(num_timesteps=10, beta_start=1e-4, beta_end=0.02)
    print(f"\n调度器参数:")
    print(f"时间步数: {scheduler.num_timesteps}")
    print(f"β范围: [{scheduler.betas[0]:.6f}, {scheduler.betas[-1]:.6f}]")
    
    # 测试前向扩散过程
    print(f"\n测试前向扩散过程:")
    timesteps = torch.tensor([5, 8])  # 选择两个不同的时间步
    noise = torch.randn_like(test_image)
    noisy_image = scheduler.add_noise(test_image, noise, timesteps)
    print(f"加噪后图像形状: {noisy_image.shape}")
    print(f"时间步 {timesteps[0]} 的累积α: {scheduler.alphas_cumprod[timesteps[0]]:.6f}")
    
    # 测试反向扩散过程
    print(f"\n测试反向扩散过程:")
    # 模拟模型输出（在实际应用中这应该是UNet的输出）
    model_output = torch.randn_like(test_image)
    timestep = 5
    denoised_image = scheduler.step(model_output, timestep, noisy_image)
    print(f"去噪后图像形状: {denoised_image.shape}")
    print(f"时间步 {timestep} 的后验方差: {scheduler.posterior_variance[timestep]:.6f}")
    
    # 验证数值范围和梯度
    print(f"\n数值验证:")
    print(f"前向过程中的最大值: {noisy_image.max():.6f}")
    print(f"前向过程中的最小值: {noisy_image.min():.6f}")
    print(f"反向过程中的最大值: {denoised_image.max():.6f}")
    print(f"反向过程中的最小值: {denoised_image.min():.6f}")
