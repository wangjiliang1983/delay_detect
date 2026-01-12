# AD/DA 链路延时测量系统设计说明书

## 1. 任务背景与目标

### 1.1 项目概述

在采样率为 1 GHz（采样周期 Ts = 1 ns）的系统中，需要测量 DA 输出到 AD 输入之间的物理链路延时。本系统旨在超越 1 ns 的原生分辨率限制，利用信号处理算法实现亚纳秒（皮秒级）的测量精度。

### 1.2 系统参数

| 参数 | 数值 | 说明 |
|------|------|------|
| 采样率 | 1 GSPS | AD/DA 采样率 |
| 采样周期 | 1 ns | 原生时间分辨率 |
| 目标精度 | < 10 ps | 亚纳秒级测量 |
| 时钟源 | 共时钟 | AD/DA 使用同一时钟源 |

### 1.3 核心目标

- 实现高线性度的自动化延时测量
- 突破 1 ns 原生采样分辨率限制
- 在噪声环境下保持稳定测量精度
- 提供可重复、可验证的测量结果

### 1.4 技术挑战

- 物理链路延时通常不是采样周期的整数倍
- 需要通过信号处理算法进行亚采样精度估计
- 时钟抖动和噪声会影响测量精度
- 需要在高线性度条件下进行自动化测量

---

## 2. 系统架构设计

### 2.1 整体架构

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   发射端    │    │   模拟信道   │    │   接收端    │
│    (TX)     │───▶│  (Channel)  │───▶│    (RX)     │
└─────────────┘    └─────────────┘    └─────────────┘
      │                                      │
      │         ┌─────────────┐              │
      └────────▶│  本地参考   │◀─────────────┘
                │  信号源     │
                └─────────────┘
```

### 2.2 发射端 (TX) 详细设计

#### 2.2.1 m 序列生成器

**技术选择**：m 序列（最大长度序列）

**参数规格**：
| 参数 | 数值 | 说明 |
|------|------|------|
| 阶数 | 10 | 决定序列长度和特性 |
| 长度 | 1023 | 2^10 - 1 |
| 生成多项式 | x^10 + x^3 + 1 | 本原多项式 |
| 码元速率 | 1 GHz | 与采样率相同 |

**m 序列特性**：
- 理想的自相关特性
- 自相关函数在零位为 N-1，其他位置为 -1
- 频谱接近白噪声，适合宽带测量

**实现算法**：
```python
def generate_m_sequence(order: int = 10, seed: int = 1) -> np.ndarray:
    """
    生成 m 序列

    Parameters:
        order: 阶数 (默认 10)
        seed: 初始状态种子 (默认 1)

    Returns:
        m 序列数组 (长度为 2^order - 1)
    """
    # 1. 初始化移位寄存器
    state = seed & ((1 << order) - 1)

    # 2. 生成多项式系数
    # x^10 + x^3 + 1 对应位: 10, 3, 0
    taps = [order, 3, 0]

    # 3. 生成序列
    sequence = []
    for _ in range((1 << order) - 1):
        # 计算输出位
        output = state & 1
        sequence.append(output)

        # 计算反馈位
        feedback = 0
        for tap in taps[:-1]:  # 排除最后一个 taps[2]=0
            if (state >> (tap - 1)) & 1:
                feedback ^= 1

        # 移位并注入反馈
        state = (state >> 1) | (feedback << (order - 1))

    return np.array(sequence)
```

#### 2.2.2 BPSK 调制器

**技术选择**：二进制相移键控 (BPSK)

**参数规格**：
| 参数 | 数值 | 说明 |
|------|------|------|
| 载波频率 | 200 MHz | 中频频率 |
| 调制方式 | BPSK | 0 → +1, 1 → -1 |
| 符号速率 | 1 Gbaud | 与采样率匹配 |
| 带宽 | ~500 MHz | m 序列基带带宽 |

**调制公式**：
```
s(n) = m(n) × cos(2π × fc / fs × n)
```

其中：
- m(n) ∈ {+1, -1}（映射后的 m 序列）
- fc = 200 MHz（载波频率）
- fs = 1000 MHz（采样频率）

**实现算法**：
```python
def bpsk_modulate(bits: np.ndarray, fc: float, fs: float) -> np.ndarray:
    """
    BPSK 调制

    Parameters:
        bits: 输入比特序列 (0 或 1)
        fc: 载波频率 (Hz)
        fs: 采样频率 (Hz)

    Returns:
        调制后的复数基带信号
    """
    # 1. 将 0/1 映射为 +1/-1
    symbols = 2 * bits.astype(np.float64) - 1

    # 2. 生成载波
    n = np.arange(len(bits))
    carrier = np.cos(2 * np.pi * fc / fs * n)

    # 3. 调制
    modulated = symbols * carrier

    return modulated
```

#### 2.2.3 发射端参数汇总

| 参数 | 数值 | 单位 |
|------|------|------|
| m 序列阶数 | 10 | - |
| m 序列长度 | 1023 | 采样点 |
| m 序列周期 | 1023 | ns |
| 载波频率 | 200 | MHz |
| 调制方式 | BPSK | - |
| 信号带宽 | 500 | MHz |
| 重复频率 | 0.977 | MHz |

### 2.3 模拟信道 (Channel) 详细设计

#### 2.3.1 分数阶时延模型

**物理延时分解**：
```
Total Delay = τ_int + τ_frac

其中：
- τ_int: 整数倍 Ts 的延时 (k × 1 ns)
- τ_frac: 分数倍 Ts 的延时 (-0.5 ~ +0.5 ns)
```

**时域模型**：
```
y(t) = x(t - τ) + n(t)

其中：
- x(t): 发射信号
- y(t): 接收信号
- τ: 物理延时
- n(t): 噪声
```

**频域模型**：
```
Y(f) = X(f) × e^(-j2πfτ) + N(f)

其中：
- H(f) = e^(-j2πfτ) 为全通滤波器
```

#### 2.3.2 频域实现方法

**实现原理**：
利用 FFT/IFFT 在频域实现分数阶延时

**算法步骤**：
1. 对输入信号 x(n) 进行 N 点 FFT
2. 对每个频点乘以相移因子 e^(-j2πfΔτ)
3. 进行 IFFT 得到延时后的信号

**实现算法**：
```python
def fractional_delay_fd(signal: np.ndarray, delay_ns: float, fs: float) -> np.ndarray:
    """
    频域分数阶延时

    Parameters:
        signal: 输入信号
        delay_ns: 延时 (ns)
        fs: 采样频率 (Hz)

    Returns:
        延时后的信号
    """
    n = len(signal)

    # 1. FFT
    X = np.fft.fft(signal)

    # 2. 生成频率轴
    freq = np.fft.fftfreq(n, 1/fs)

    # 3. 计算相移
    phase_shift = np.exp(-1j * 2 * np.pi * freq * delay_ns * 1e-9)

    # 4. 频域相移
    Y = X * phase_shift

    # 5. IFFT
    y = np.fft.ifft(Y)

    return np.real(y)
```

#### 2.3.3 时域实现方法（可选）

**实现原理**：
使用 Lagrange 插值滤波器进行分数阶延时

**Farrow 结构**：
适用于需要动态调整延时的场景

**实现算法**：
```python
def fractional_delay_lagrange(signal: np.ndarray, delay_samples: float) -> np.ndarray:
    """
    时域 Lagrange 插值分数阶延时

    Parameters:
        signal: 输入信号
        delay_samples: 延时 (采样点，可为小数)

    Returns:
        延时后的信号
    """
    n = len(signal)
    integer_delay = int(np.floor(delay_samples))
    fractional_delay = delay_samples - integer_delay

    # 使用 4 阶 Lagrange 插值
    output = np.zeros(n)

    for i in range(n):
        if i - integer_delay - 1 < 0 or i - integer_delay + 2 >= n:
            output[i] = signal[i]
        else:
            x0 = signal[i - integer_delay - 1]
            x1 = signal[i - integer_delay]
            x2 = signal[i - integer_delay + 1]
            x3 = signal[i - integer_delay + 2]

            # 4 阶 Lagrange 插值
            t = fractional_delay
            output[i] = (
                (-t * (t - 1) * (t - 2) / 6) * x0 +
                ((t + 1) * (t - 1) * (t - 2) / 2) * x1 +
                (-t * (t + 1) * (t - 2) / 2) * x2 +
                (t * (t + 1) * (t - 1) / 6) * x3
            )

    return output
```

### 2.4 接收端 (RX) 详细设计

#### 2.4.1 互相关器

**技术原理**：
利用互相关运算识别接收信号与参考信号的时间对齐位置

**数学公式**：
```
R_xy[k] = Σ_{n=0}^{N-1} y[n] × x*[n - k]

其中：
- y[n]: 接收信号
- x[n]: 本地参考信号
- *: 复共轭
```

**互相关特性**：
- 当 x 和 y 完全对齐时，互相关值最大
- 峰值位置对应信号延时
- 互相关提供约 N 倍的信噪比增益（N 为序列长度）

**实现算法**：
```python
def cross_correlation(sig1: np.ndarray, sig2: np.ndarray) -> np.ndarray:
    """
    互相关计算

    Parameters:
        sig1: 信号 1
        sig2: 信号 2

    Returns:
        互相关结果
    """
    # 使用 FFT 实现快速互相关
    n = len(sig1) + len(sig2) - 1

    # FFT
    F1 = np.fft.fft(sig1, n)
    F2 = np.fft.fft(sig2, n)

    # 频域相乘（F1 共轭 * F2）
    corr_fft = np.conj(F1) * F2

    # IFFT
    correlation = np.fft.ifft(corr_fft)

    return np.real(correlation)
```

#### 2.4.2 峰值搜索

**实现步骤**：
1. 找到互相关结果的最大值位置
2. 该位置对应整数倍采样周期的延时

**实现算法**：
```python
def find_peak_index(correlation: np.ndarray) -> int:
    """
    查找互相关峰值位置

    Parameters:
        correlation: 互相关结果

    Returns:
        峰值索引
    """
    return np.argmax(np.abs(correlation))
```

#### 2.4.3 抛物线插值

**技术原理**：
利用峰值及其左右相邻点的幅值进行二次函数拟合，精确估计真实极值点位置

**数学推导**：

设二次函数为：
```
y(t) = a × t² + b × t + c
```

其中 t = -1, 0, 1 分别对应 left, peak, right 三个采样点

求解偏移量：
```
δ = (y_left - y_right) / [2(y_left - 2y_peak + y_right)]

其中：
- y_left: 峰值左侧采样点幅值
- y_peak: 峰值点幅值
- y_right: 峰值右侧采样点幅值
- δ: 相对于峰值中心的偏移量 (-0.5 ~ +0.5)
```

**实现算法**：
```python
def parabolic_interpolation(y_left: float, y_peak: float, y_right: float) -> float:
    """
    抛物线插值估计峰值偏移

    Parameters:
        y_left: 峰值左侧点幅值 (索引 -1)
        y_peak: 峰值点幅值 (索引 0)
        y_right: 峰值右侧点幅值 (索引 +1)

    Returns:
        偏移量 δ (-0.5 ~ +0.5 采样周期)
    """
    numerator = y_left - y_right
    denominator = 2 * (y_left - 2 * y_peak + y_right)

    if np.abs(denominator) < 1e-10:
        return 0.0

    delta = numerator / denominator

    # 限制范围
    delta = np.clip(delta, -0.5, 0.5)

    return delta
```

#### 2.4.4 延时估计器

**完整流程**：

```
接收信号 y(n)
      │
      ▼
┌──────────────────┐
│  生成本地参考信号 │◀─── m 序列 + BPSK
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│    互相关运算     │────▶ R_xy[k]
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   峰值搜索       │────▶ k_peak
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  抛物线插值      │────▶ δ
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 延时 = (k_peak + δ) × Ts
└──────────────────┘
```

**实现算法**：
```python
def estimate_delay(rx_signal: np.ndarray,
                   tx_signal: np.ndarray,
                   fs: float) -> float:
    """
    估计信号延时

    Parameters:
        rx_signal: 接收信号
        tx_signal: 发射信号（参考信号）
        fs: 采样频率 (Hz)

    Returns:
        估计延时 (秒)
    """
    # 1. 互相关
    correlation = cross_correlation(rx_signal, tx_signal)

    # 2. 峰值搜索
    k_peak = find_peak_index(correlation)

    # 3. 提取峰值附近三点
    n = len(correlation)

    # 处理边界情况
    if k_peak == 0:
        y_left = 0
        y_peak = correlation[0]
        y_right = correlation[1]
    elif k_peak == n - 1:
        y_left = correlation[n - 2]
        y_peak = correlation[n - 1]
        y_right = 0
    else:
        y_left = correlation[k_peak - 1]
        y_peak = correlation[k_peak]
        y_right = correlation[k_peak + 1]

    # 4. 抛物线插值
    delta = parabolic_interpolation(y_left, y_peak, y_right)

    # 5. 计算总延时
    ts = 1 / fs
    total_delay = (k_peak + delta) * ts

    return total_delay
```

---

## 3. 性能评估指标 (KPI)

### 3.1 精度指标

| 指标 | 要求 | 说明 |
|------|------|------|
| 测量精度 | < 10 ps | 均方根误差 (RMSE) |
| 分辨率 | 1 ps | 理论最小分辨能力 |
| 线性度 | < 0.1% | 延时测量偏差随真实延时变化的线性程度 |
| 重复性 | < 5 ps | 多次测量结果的标准差 |
| 偏移量 | < 5 ps | 系统性偏差（可校准） |

### 3.2 性能指标

| 指标 | 要求 | 说明 |
|------|------|------|
| 信噪比增益 | > 30 dB | 互相关处理后的 SNR 改善 |
| 测量范围 | 0 ~ 10 μs | 可测量的延时范围 |
| 测量速度 | > 1000 次/秒 | 自动化测量吞吐率 |
| 动态范围 | > 60 dB | 可测量的信号功率范围 |

### 3.3 误差来源分析

| 误差源 | 影响程度 | 缓解措施 |
|--------|----------|----------|
| 量化噪声 | 中 | 增加量化位数（14-bit 以上） |
| 时钟抖动 | 高 | 使用 < 100 fs 抖动的高精度时钟 |
| 频率偏移 | 中 | 载波同步或使用零中频方案 |
| 多径效应 | 中 | 宽带信号设计、通道校准 |
| 滤波器非线性 | 中 | 预失真校正、线性相位滤波器 |
| 热噪声 | 低 | 互相关平均增加 SNR |
| 采样时钟偏移 | 高 | 使用共用高精度时钟源 |

---

## 4. 仿真测试方案

### 4.1 延时扫描测试

#### 4.1.1 测试配置

| 参数 | 数值 | 说明 |
|------|------|------|
| 延时范围 | 100.0 ns ~ 105.0 ns | 扫描范围 |
| 步进间隔 | 0.05 ns | 扫描分辨率 |
| 测试点数 | 101 | 延时点数量 |
| 每点测量次数 | 100 | 统计平均 |
| 噪声条件 | 可选 | SNR = 20 dB |

#### 4.1.2 测试流程

```
开始测试
    │
    ▼
┌─────────────────────────────────────┐
│ 1. 设定真实延时值                   │
│    从 100.0 ns 步进到 105.0 ns      │
└───────────────┬─────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│ 2. 生成测试信号                     │
│    - m 序列 + BPSK 调制             │
│    - 长度 = 10230 采样点 (10 周期)  │
└───────────────┬─────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│ 3. 应用分数阶延时                   │
│    - τ = τ_real + τ_noise           │
│    - τ_noise ~ N(0, 0.01 ns)        │
└───────────────┬─────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│ 4. 接收端处理                       │
│    - 互相关运算                     │
│    - 峰值搜索                       │
│    - 抛物线插值                     │
└───────────────┬─────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│ 5. 记录测量结果                     │
│    - 测量延时                       │
│    - 计算误差                       │
└───────────────┬─────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│ 6. 重复测量 100 次                  │
│    计算均值和标准差                 │
└───────────────┬─────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│ 7. 步进到下一个延时点               │
│    重复步骤 2-6                     │
└───────────────┬─────────────────────┘
                │
                ▼
         所有点测试完成？
                │
           是   │
                ▼
         ┌─────────────────┐
         │ 8. 生成测试报告 │
         │    - RMSE       │
         │    - 误差曲线   │
         │    - 统计指标   │
         └─────────────────┘
```

#### 4.1.3 评估项目

| 评估项目 | 计算方法 | 要求 |
|----------|----------|------|
| 均方根误差 (RMSE) | sqrt(mean((τ_meas - τ_true)^2)) | < 10 ps |
| 最大绝对误差 | max(abs(τ_meas - τ_true)) | < 20 ps |
| 平均偏差 | mean(τ_meas - τ_true) | < 5 ps |
| 误差标准差 | std(τ_meas - τ_true) | < 5 ps |

### 4.2 噪声鲁棒性测试

#### 4.2.1 测试配置

| 参数 | 数值 | 说明 |
|------|------|------|
| SNR 范围 | -20 dB ~ 20 dB | 信噪比扫描 |
| 步进 | 5 dB | 扫描间隔 |
| 固定延时 | 100.3 ns | 延时真值 |
| 测量次数 | 100 | 每 SNR 点 |

#### 4.2.2 测试目的

验证算法在不同噪声条件下的测量精度，绘制 SNR vs. RMSE 曲线。

### 4.3 线性度测试

#### 4.3.1 测试配置

| 参数 | 数值 | 说明 |
|------|------|------|
| 延时范围 | 0 ~ 1000 ns | 大范围扫描 |
| 步进 | 1 ns | 扫描间隔 |
| 测量次数 | 10 | 每点测量 |

#### 4.3.2 测试目的

评估测量误差随延时真值变化的规律，验证系统线性度。

---

## 5. Python 仿真代码实现

### 5.1 模块结构

```
adda/
├── Design.md              # 设计文档（本文档）
├── README.md              # 项目说明
├── requirements.txt       # Python 依赖
├── setup.py               # 安装配置
├── simulation/
│   ├── __init__.py
│   ├── tx.py              # 发射端模块
│   │   ├── generate_m_sequence()
│   │   └── bpsk_modulate()
│   ├── channel.py         # 信道模块
│   │   ├── fractional_delay_fd()
│   │   └── fractional_delay_lagrange()
│   ├── rx.py              # 接收端模块
│   │   ├── cross_correlation()
│   │   ├── find_peak_index()
│   │   ├── parabolic_interpolation()
│   │   └── estimate_delay()
│   ├── metrics.py         # 评估指标模块
│   │   ├── calculate_rmse()
│   │   ├── calculate_max_error()
│   │   ├── calculate_bias()
│   │   └── calculate_std()
│   └── utils.py           # 工具函数
│       ├── add_awgn()
│       └── normalize()
├── tests/
│   ├── __init__.py
│   ├── test_tx.py         # 发射端测试
│   ├── test_channel.py    # 信道测试
│   ├── test_rx.py         # 接收端测试
│   └── test_integration.py # 集成测试
└── examples/
    ├── delay_sweep.py     # 延时扫描示例
    ├── noise_test.py      # 噪声鲁棒性测试
    └── demo.py            # 演示脚本
```

### 5.2 核心函数接口

#### 5.2.1 发射端模块 (tx.py)

```python
import numpy as np
from typing import Optional

def generate_m_sequence(order: int = 10, seed: int = 1) -> np.ndarray:
    """
    生成 m 序列（最大长度序列）

    Parameters:
        order: 阶数，默认 10
        seed: 初始状态种子，默认 1

    Returns:
        m 序列数组，长度为 2^order - 1
    """

def bpsk_modulate(bits: np.ndarray, fc: float, fs: float) -> np.ndarray:
    """
    BPSK 调制

    Parameters:
        bits: 输入比特序列 (0 或 1)
        fc: 载波频率 (Hz)
        fs: 采样频率 (Hz)

    Returns:
        调制后的实数信号
    """

def generate_tx_signal(order: int = 10,
                       fc: float = 200e6,
                       fs: float = 1e9,
                       num_periods: int = 10) -> np.ndarray:
    """
    生成完整发射信号

    Parameters:
        order: m 序列阶数
        fc: 载波频率 (Hz)
        fs: 采样频率 (Hz)
        num_periods: m 序列周期数

    Returns:
        发射信号数组
    """
```

#### 5.2.2 信道模块 (channel.py)

```python
import numpy as np

def fractional_delay_fd(signal: np.ndarray,
                        delay_ns: float,
                        fs: float) -> np.ndarray:
    """
    频域分数阶延时

    Parameters:
        signal: 输入信号
        delay_ns: 延时 (ns)
        fs: 采样频率 (Hz)

    Returns:
        延时后的信号
    """

def fractional_delay_lagrange(signal: np.ndarray,
                              delay_samples: float) -> np.ndarray:
    """
    时域 Lagrange 插值分数阶延时

    Parameters:
        signal: 输入信号
        delay_samples: 延时 (采样点，可为小数)

    Returns:
        延时后的信号
    """

def add_awgn(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """
    添加高斯白噪声

    Parameters:
        signal: 输入信号
        snr_db: 信噪比 (dB)

    Returns:
        加噪后的信号
    """
```

#### 5.2.3 接收端模块 (rx.py)

```python
import numpy as np

def cross_correlation(sig1: np.ndarray, sig2: np.ndarray) -> np.ndarray:
    """
    互相关计算（使用 FFT 加速）

    Parameters:
        sig1: 信号 1
        sig2: 信号 2

    Returns:
        互相关结果
    """

def find_peak_index(correlation: np.ndarray) -> int:
    """
    查找互相关峰值位置

    Parameters:
        correlation: 互相关结果

    Returns:
        峰值索引
    """

def parabolic_interpolation(y_left: float,
                            y_peak: float,
                            y_right: float) -> float:
    """
    抛物线插值估计峰值偏移

    Parameters:
        y_left: 峰值左侧点幅值 (索引 -1)
        y_peak: 峰值点幅值 (索引 0)
        y_right: 峰值右侧点幅值 (索引 +1)

    Returns:
        偏移量 δ (-0.5 ~ +0.5 采样周期)
    """

def estimate_delay(rx_signal: np.ndarray,
                   tx_signal: np.ndarray,
                   fs: float) -> float:
    """
    估计信号延时

    Parameters:
        rx_signal: 接收信号
        tx_signal: 发射信号（参考信号）
        fs: 采样频率 (Hz)

    Returns:
        估计延时 (秒)
    """
```

#### 5.2.4 评估指标模块 (metrics.py)

```python
import numpy as np

def calculate_rmse(measured: np.ndarray, actual: np.ndarray) -> float:
    """
    计算均方根误差 (RMSE)

    Parameters:
        measured: 测量值数组
        actual: 真实值数组

    Returns:
        RMSE 值
    """

def calculate_max_error(measured: np.ndarray, actual: np.ndarray) -> float:
    """
    计算最大绝对误差

    Parameters:
        measured: 测量值数组
        actual: 真实值数组

    Returns:
        最大绝对误差
    """

def calculate_bias(measured: np.ndarray, actual: np.ndarray) -> float:
    """
    计算平均偏差

    Parameters:
        measured: 测量值数组
        actual: 真实值数组

    Returns:
        平均偏差
    """

def calculate_std(measured: np.ndarray, actual: np.ndarray) -> float:
    """
    计算误差标准差

    Parameters:
        measured: 测量值数组
        actual: 真实值数组

    Returns:
        标准差
    """
```

### 5.3 示例代码

#### 5.3.1 延时扫描示例 (examples/delay_sweep.py)

```python
"""
延时扫描测试示例

测试不同延时值下的测量精度
"""

import numpy as np
import matplotlib.pyplot as plt
from simulation import (
    generate_tx_signal,
    fractional_delay_fd,
    estimate_delay,
    add_awgn
)
from simulation.metrics import (
    calculate_rmse,
    calculate_max_error,
    calculate_bias,
    calculate_std
)

def run_delay_sweep():
    """运行延时扫描测试"""

    # 参数设置
    fs = 1e9  # 采样率 1 GHz
    fc = 200e6  # 载波频率 200 MHz
    num_periods = 10  # 10 个 m 序列周期

    # 延时扫描参数
    delay_start = 100.0  # ns
    delay_stop = 105.0  # ns
    delay_step = 0.05  # ns
    snr_db = 20  # 信噪比

    # 生成发射信号
    tx_signal = generate_tx_signal(fc=fc, fs=fs, num_periods=num_periods)

    # 延时扫描
    delays_true = np.arange(delay_start, delay_stop + delay_step, delay_step)
    delays_measured = []

    for delay_ns in delays_true:
        # 应用延时
        rx_signal = fractional_delay_fd(tx_signal, delay_ns, fs)

        # 添加噪声
        rx_signal_noisy = add_awgn(rx_signal, snr_db)

        # 估计延时
        delay_est = estimate_delay(rx_signal_noisy, tx_signal, fs) * 1e9  # 转换为 ns
        delays_measured.append(delay_est)

    delays_measured = np.array(delays_measured)

    # 计算评估指标
    errors = (delays_measured - delays_true) * 1e3  # 转换为 ps

    rmse = calculate_rmse(delays_measured * 1e-9, delays_true * 1e-9) * 1e12
    max_error = calculate_max_error(delays_measured * 1e-9, delays_true * 1e-9) * 1e12
    bias = calculate_bias(delays_measured * 1e-9, delays_true * 1e-9) * 1e12
    std = calculate_std(delays_measured * 1e-9, delays_true * 1e-9) * 1e12

    # 打印结果
    print(f"延时扫描测试结果 (SNR = {snr_db} dB)")
    print(f"  RMSE: {rmse:.2f} ps")
    print(f"  最大误差: {max_error:.2f} ps")
    print(f"  偏差: {bias:.2f} ps")
    print(f"  标准差: {std:.2f} ps")

    # 绘制误差曲线
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(delays_true, errors, 'b-', linewidth=1)
    plt.xlabel('True Delay (ns)')
    plt.ylabel('Error (ps)')
    plt.title('Delay Measurement Error')
    plt.grid(True)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)

    plt.subplot(2, 1, 2)
    plt.hist(errors, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Error (ps)')
    plt.ylabel('Count')
    plt.title('Error Distribution')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('delay_sweep_results.png', dpi=150)
    plt.show()

    return {
        'delays_true': delays_true,
        'delays_measured': delays_measured,
        'errors': errors,
        'rmse': rmse,
        'max_error': max_error,
        'bias': bias,
        'std': std
    }

if __name__ == '__main__':
    results = run_delay_sweep()
```

---

## 6. 硬件实现注意事项

### 6.1 时钟抖动 (Jitter)

**影响分析**：
- 时钟抖动会在高频处引入相位噪声
- 对于 200 MHz 载波，时钟抖动会直接影响相位测量精度

**缓解措施**：
| 参数 | 要求 | 说明 |
|------|------|------|
| 时钟抖动 | < 100 fs | RMS 抖动 |
| 时钟稳定性 | < 1 ppm | 频率稳定性 |
| 上升时间 | < 100 ps | 时钟边沿质量 |

**抖动影响计算**：
```
相位噪声 = 2π × f × τ_jitter

其中：
- f = 200 MHz（信号频率）
- τ_jitter = 100 fs（时钟抖动）

相位误差 = 2π × 200e6 × 100e-15 = 0.000126 弧度
等效时间误差 = 0.000126 / (2π × 200e6) ≈ 0.1 ps
```

### 6.2 阻抗匹配

**影响分析**：
- 不匹配会导致信号反射
- 反射信号会叠加在主信号上造成干涉

**缓解措施**：
- 确保 50 Ω 阻抗匹配
- 使用端接电阻
- 控制 PCB 走线阻抗

### 6.3 滤波器设计

**要求**：
- 抗混叠滤波器需要有线性相位响应
- 避免相位畸变影响测量精度

**推荐规格**：
| 参数 | 要求 | 说明 |
|------|------|------|
| 通带平坦度 | < 0.1 dB | 幅度响应一致性 |
| 群时延波动 | < 1 ns | 相位响应线性度 |
| 阻带衰减 | > 60 dB | 杂散抑制 |

### 6.4 校准流程

**系统校准步骤**：
1. 零延时校准：测量已知短电缆的延时作为基准
2. 增益校准：校准 AD/DA 增益一致性
3. 相位校准：校准通道间相位一致性
4. 线性度校准：建立校准曲线校正系统性偏差

**定期校准**：
- 建议每周进行一次完整校准
- 环境温度变化较大时需重新校准

---

## 7. 结论

### 7.1 设计总结

本设计方案基于以下核心技术实现亚纳秒级延时测量：

| 技术 | 作用 | 效果 |
|------|------|------|
| m 序列 (10阶) | 提供宽带特征信号 | 理想自相关特性 |
| BPSK 调制 | 频谱搬移 | 抗低频干扰 |
| 互相关运算 | 时间对齐检测 | SNR 增益 > 30 dB |
| 抛物线插值 | 亚采样精度估计 | 分辨率 < 1 ps |

### 7.2 性能指标预期

| 指标 | 预期值 | 备注 |
|------|--------|------|
| 测量精度 | < 10 ps | 均方根误差 |
| 分辨率 | 1 ps | 理论极限 |
| 线性度 | < 0.1% | 大范围测量 |
| 重复性 | < 5 ps | 多次测量一致 |

### 7.3 注意事项

实际硬件测试中，如发现误差大于仿真预期，应重点排查：

1. **时钟抖动**：使用 < 100 fs 抖动的高精度时钟源
2. **阻抗匹配**：确保 50 Ω 匹配，减少反射
3. **滤波器非线性**：使用线性相位滤波器
4. **电源噪声**：良好的电源滤波和接地
5. **温度漂移**：控制环境温度稳定

### 7.4 后续工作

- 实现完整的 Python 仿真代码
- 编写单元测试和集成测试
- 进行仿真验证
- 硬件平台移植和实测验证
- 优化算法性能和实时性

---

*文档版本：1.0*
*创建日期：2025年1月*
