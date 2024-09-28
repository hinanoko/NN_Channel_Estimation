from __future__ import division
import numpy as np
import scipy.interpolate
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import math
import os

carrier = 64
num_symbol = 50
CP = 8
pilot_Inter = 8   # 导航符号间隔
bitNum = 2
SNRdb = 20

# 生成随机比特数列
bit_length = num_symbol * carrier
bit_sequence = np.random.randint(0, 4, size=bit_length)

# 设置打印选项以查看完整数组
np.set_printoptions(threshold=np.inf)  # 设置为无限制显示
print(bit_sequence)

# 定义映射字典
symbol_mapping = {
    0: -1 - 1j,  # 对应0
    1: -1 + 1j,  # 对应1
    2: 1 + 1j,   # 对应2
    3: 1 - 1j    # 对应3
}

# 使用映射字典进行符号调制
modulated_symbols = np.array([symbol_mapping[bit] for bit in bit_sequence])

# 打印调制后的符号
print("调制后的符号:", modulated_symbols)

num_pilot = int(np.ceil(num_symbol / pilot_Inter))
num_data = num_symbol + num_pilot  # 包括导航符号和数据符号

# 初始化 pilot_Indx 和 Data_Indx
pilot_Indx = np.zeros(num_pilot, dtype=int)
Data_Indx = np.zeros(num_pilot * (pilot_Inter + 1), dtype=int)

# 重新计算 num_pilot 和 num_data
num_pilot = int(np.ceil(num_symbol / pilot_Inter))
num_data = num_symbol + num_pilot

# 初始化 pilot_Indx 和 Data_Indx
pilot_Indx = np.zeros(num_pilot, dtype=int)

# 正确计算导频索引，确保从0开始，并在 [0, 9, 18, 27, 36, 45, 56] 插入导频
for i in range(num_pilot):
    pilot_Indx[i] = i * (pilot_Inter + 1)

# 如果最后的导频索引超过了 num_data，需要修正索引，确保不会超过符号数
if pilot_Indx[-1] >= num_data:
    pilot_Indx[-1] = num_data - 1

# 计算 Data_Indx
for j in range(num_pilot):
    start = j * pilot_Inter
    Data_Indx[start:(start + pilot_Inter)] = np.arange(1 + j * (pilot_Inter + 1), (j + 1) * (pilot_Inter + 1))

# 截断 Data_Indx 以匹配 num_symbol
Data_Indx = Data_Indx[:num_symbol]

# 确保 Data_Indx 不超出范围
Data_Indx = np.clip(Data_Indx, 0, num_data - 1)

# 打印 pilot_Indx 和 Data_Indx 以检查结果
print("pilot_Indx:", pilot_Indx)
print("Data_Indx:", Data_Indx)

# 随机生成 0-3 之间的值，并映射到 QAM 调制作为导频符号
pilot_symbols = np.array([symbol_mapping[x] for x in np.random.randint(0, 4, carrier)])

# 打印导频符号
print("导频符号:", pilot_symbols)

# Step 2: 初始化矩阵
piloted_ofdm_syms = np.zeros((carrier, num_data), dtype=complex)

# Step 3: 插入调制序列到数据索引位置
Modulated_Sequence = np.random.randint(0, 4, carrier * num_symbol)  # 假设有一个调制后的序列
Modulated_Sequence_reshaped = np.reshape([symbol_mapping[bit] for bit in Modulated_Sequence], (carrier, num_symbol))
piloted_ofdm_syms[:, Data_Indx] = Modulated_Sequence_reshaped

# Step 4: 插入导频符号到导航索引位置
piloted_ofdm_syms[:, pilot_Indx] = np.tile(pilot_symbols, (num_pilot, 1)).T

# 检查矩阵形状是否正确
print("\n最终 OFDM 矩阵的形状:", piloted_ofdm_syms.shape)

# 打印最终结果矩阵
np.set_printoptions(precision=2, suppress=True)
print("\n导频和数据混合后的 OFDM 矩阵:\n", piloted_ofdm_syms)

# Step 5: 进行 IFFT 转换
time_signal = np.sqrt(carrier) * np.fft.ifft(piloted_ofdm_syms, axis=0)

# 打印转换后的时域信号
print("\n时域信号 (Time Domain Signal):\n", time_signal)

# 从 time_signal 的最后 8 个样本中提取作为循环前缀
cyclic_prefix = time_signal[-CP:, :]

# 将循环前缀添加到 time_signal 的前面
add_cyclic_signal = np.vstack((cyclic_prefix, time_signal))

# 打印最终的信号带有循环前缀
print("\n带有循环前缀的时域信号 (Time Domain Signal with Cyclic Prefix):\n", add_cyclic_signal)

print("带有循环前缀的时域信号的形状:", add_cyclic_signal.shape)
# 输出应该是 (72, 50)

