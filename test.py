from __future__ import division
import numpy as np
import scipy.interpolate
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import math
import os
import random

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
# 输出应该是 (72, 57)

# 计算总的数目
num_data = add_cyclic_signal.shape[1]  # 57

# 进行 reshape 操作
Tx_data_trans = np.reshape(add_cyclic_signal, (1, (carrier + CP) * num_data))

# 打印最终的形状
print("展平后的数据形状:", Tx_data_trans.shape)

H_folder_train = 'H_dataset_fewer/'
channel_response_set_train = []
train_idx_low = 1
train_idx_high = 21
for train_idx in range(train_idx_low, train_idx_high):
    print("Processing the ", train_idx, "th document")
    H_file = H_folder_train + str(train_idx) + '.txt'
    with open(H_file) as f:
        for line in f:
            numbers_str = line.split()
            numbers_float = [float(x) for x in numbers_str]
            h_response = np.asarray(numbers_float[0:int(len(numbers_float) / 2)]) + 1j * np.asarray(
                numbers_float[int(len(numbers_float) / 2):len(numbers_float)])
            channel_response_set_train.append(h_response)

# 随机选择60个信道响应
selected_channels = random.sample(channel_response_set_train, 60)

# 创建一个长度为960的冲激响应序列
long_impulse_response = np.concatenate(selected_channels)

# 验证长度
print(f"Length of the long impulse response: {len(long_impulse_response)}")

Tx_data_trans = Tx_data_trans.flatten()  # 或者 Tx_data_trans.reshape(-1)

print(Tx_data_trans.shape)
print(long_impulse_response.shape)

convolved = np.convolve(Tx_data_trans, long_impulse_response)
signal_power = np.mean(abs(convolved**2))
sigma2 = signal_power * 10**(-SNRdb/10)
noise = np.sqrt(sigma2/2) * (np.random.randn(*convolved.shape)+1j*np.random.randn(*convolved.shape))

OFDM_RX = convolved + noise

print(OFDM_RX.shape)

# Step 1: 丢弃卷积后信号的前 959 个样本
effective_OFDM_RX = OFDM_RX[959:]  # 丢弃前 959 个卷积引入的多余部分

# Step 2: 计算新的符号数量和每个符号的总长度
total_symbol_length = carrier + CP  # 每个符号的长度为 72
num_ofdm_symbols = len(effective_OFDM_RX) // total_symbol_length  # 计算符号数量

# Step 3: 初始化去除循环前缀后的矩阵
OFDM_RX_no_prefix = np.zeros((carrier, num_ofdm_symbols), dtype=complex)  # 64 x 57

# Step 4: 去除循环前缀并提取有效数据部分
for i in range(num_ofdm_symbols):
    start = i * total_symbol_length + CP  # 跳过每个符号的前 8 个循环前缀
    end = (i + 1) * total_symbol_length  # 每个符号的结束位置
    OFDM_RX_no_prefix[:, i] = effective_OFDM_RX[start:end]  # 提取每个符号的有效数据部分

# Step 5: 打印去掉循环前缀后的 OFDM 矩阵
print("\n去除循环前缀后的 OFDM 矩阵:\n", OFDM_RX_no_prefix)
print("去除循环前缀后的 OFDM 矩阵的形状:", OFDM_RX_no_prefix.shape)  # 应该是 (64, 57)


