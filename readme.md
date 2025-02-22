# Diffusion 图像生成项目

## 项目概述
本项目实现了一个基于扩散模型（Diffusion Model）的图像生成系统。通过逐步去噪的过程，模型能够生成高质量的图像。

## 安装说明

### 环境要求
- Python 3.7+
- PyTorch 1.7+
- CUDA（推荐用于GPU加速）

### 安装步骤
1. 克隆项目代码
```bash
git clone [项目地址]
cd diffusion_re
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

## 使用方法

### 训练模型
```bash
python train.py
```

### 生成图像
[待补充]

## 模型架构

### 整体架构
项目使用了U-Net架构作为骨干网络，包含以下主要组件：
- 编码器（Encoder）
- 解码器（Decoder）
- 时间编码模块
- 残差连接

### 时间步长编码
模型使用正弦位置编码来处理时间信息：

- 输入：时间步长张量
- 输出：时间步长的正弦位置嵌入
- 实现：使用正弦和余弦函数生成不同频率的波形
- 维度：可配置的嵌入维度

#### 数学原理
时间步长编码采用正弦位置编码（Sinusoidal Position Embeddings）的方法，其数学表达式为：

PE(t, 2i) = sin(t / 10000^(2i/d))
PE(t, 2i+1) = cos(t / 10000^(2i/d))

其中：
- t 是时间步长
- i 是维度的索引
- d 是嵌入维度
- PE(t,k) 表示时间t在第k维的编码值

#### 实现细节
时间编码的计算过程：
1. 将输入维度d分为两半，分别用于sin和cos计算
2. 计算基础频率：ω = ln(10000)/(d/2-1)
3. 生成不同频率的波形：f(i) = exp(-ω * i)，i∈[0, d/2)
4. 将时间信息t与频率信息结合：t * f(i)
5. 分别计算sin(t * f(i))和cos(t * f(i))
6. 连接sin和cos得到最终的d维位置嵌入

#### 优势
1. 唯一性：不同时间步长会得到唯一的编码
2. 平滑性：相近的时间步长会得到相似的编码
3. 有界性：所有编码值都在[-1, 1]范围内
4. 可扩展性：支持任意长度的时间序列

## 训练细节
[待补充]

## 实验结果
[待补充]

## 参考文献
[待补充]

## 贡献指南
[待补充]

## 许可证
[待补充]