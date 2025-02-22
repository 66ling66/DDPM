import torch
import numpy as np

class DDPMScheduler:
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.num_timesteps = num_timesteps
        
        # 定义噪声调度
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.]), self.alphas_cumprod[:-1]])
        
        # 计算扩散过程中的系数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1 / self.alphas)
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
    
    def add_noise(self, x_0, noise, timesteps):
        """前向扩散过程"""
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[timesteps].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[timesteps].reshape(-1, 1, 1, 1)
        
        # 计算带噪声的图像
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t
    
    def step(self, model_output, timestep, sample):
        """反向扩散过程中的单步去噪"""
        t = timestep
        
        # 计算去噪系数
        alpha_t = self.alphas[t]
        alpha_t_cumprod = self.alphas_cumprod[t]
        beta_t = self.betas[t]
        
        # 计算预测的x_0
        pred_original_sample = (sample - beta_t / torch.sqrt(1 - alpha_t_cumprod) * model_output) / torch.sqrt(alpha_t)
        
        # 计算方差
        variance = 0 if t == 0 else self.posterior_variance[t]
        std = torch.sqrt(variance)
        
        # 如果不是最后一步，添加噪声
        if t > 0:
            noise = torch.randn_like(model_output)
        else:
            noise = 0
            
        # 计算x_{t-1}
        pred_prev_sample = torch.sqrt(alpha_t) * pred_original_sample + \
                          torch.sqrt(1 - alpha_t - variance) * model_output + \
                          std * noise
                          
        return pred_prev_sample