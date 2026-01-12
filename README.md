# AD/DA 链路延时测量系统

基于 m 序列和互相关算法的亚纳秒级延时测量系统。

## 功能

- 发射机：生成 10 阶 m 序列并进行 BPSK 调制
- 信道：模拟分数阶延时和高斯白噪声
- 接收机：互相关 + 抛物线插值估计延时

## 性能指标

- 采样率：1 GHz
- 测量精度：~40 ps RMSE
- m 序列长度：1023 (10 阶)

## 文件结构

```
adda/
├── tx.py          # 发射机模块
├── channel.py     # 信道模块
├── rx.py          # 接收机模块
├── simulation.py  # 仿真主程序
├── Design.md      # 设计文档
└── README.md      # 本文件
```

## 运行仿真

```bash
python simulation.py
```

## 使用方法

```python
from tx import tx
from channel import channel
from rx import rx

# 参数设置
fs = 1e9  # 采样率 1 GHz
fc = 200e6  # 载波频率 200 MHz
delay_true = 100.3  # 延时真值 (ns)

# 生成发射信号
tx_signal = tx(num_periods=10, fc=fc, fs=fs)

# 通过信道
rx_signal = channel(tx_signal, delay_true, fs, snr_db=20)

# 估计延时
delay_measured = rx(rx_signal, tx_signal, fs)
print(f"测量延时: {delay_measured * 1e9:.3f} ns")
```
