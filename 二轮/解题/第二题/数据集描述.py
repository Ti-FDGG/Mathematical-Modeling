import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 模值图
acT1_complex = np.load(r"C:\Users\Timothy\Desktop\数学建模相关\二轮\数据集\data_amplitude_T1_complex.npy")
modulus = np.abs(acT1_complex)
phase = np.angle(acT1_complex)

plt.figure(figsize=(10, 6))
plt.contourf(modulus)
plt.title("data_amplitude_T1模值")
plt.xlabel("脉冲数")
plt.ylabel("距离")
plt.colorbar()
plt.show()

# 幅角图
phase1 = phase[435:460, 0:50]

# 创建x和y的值
x = np.arange(phase1.shape[1])
y = np.arange(phase1.shape[0])

# 创建一个二维网格
X, Y = np.meshgrid(x, y)

# 根据角度创建向量
U = np.cos(phase1)
V = np.sin(phase1)

# 绘制向量场图，所有的箭头都有相同的长度
plt.figure(figsize=(10, 6))
plt.quiver(X, Y, U, V)
plt.title("data_amplitude_T1相位(435:460, 0:50)")
plt.xlabel("脉冲数")
plt.ylabel("距离")
# 显示图像
plt.show()