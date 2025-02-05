import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

# 参数设置
n = 200
f_p = 0.1
alpha = 0.0081
gamma = 3.3
hmax = 2.8
kfix = 0.3 # 修正系数
# 定义JONSWAP模型
def JONSWAP(f, f_p, alpha, gamma):
    sigma = np.where(f <= f_p, 0.07, 0.09)
    r = np.exp(- (f - f_p)**2 / (2 * sigma**2 * f_p**2))
    return alpha * f**-5 * np.exp(-5/4 * (f_p / f)**4) * gamma**r

f1 = np.linspace(0.01, 2, n)
f2 = np.linspace(0.01, 2, n)
# 定义蒙特卡洛模拟
def monte_carlo_simulation(n, f_p, alpha, gamma):
    F1, F2 = np.meshgrid(f1, f2)
    spectrum = JONSWAP(np.sqrt(F1**2 + F2**2), f_p, alpha, gamma)
    phase = np.random.uniform(0, 2*np.pi, (n, n))
    return F1, F2, spectrum, phase

# 进行蒙特卡洛模拟
F1, F2, spectrum, phase = monte_carlo_simulation(n, f_p, alpha, gamma)

# 将spectrum转换为DataFrame
spectrum_df = pd.DataFrame(spectrum, index=F1[0, :], columns=F2[:, 0])
# print(spectrum)
k = kfix * hmax/np.sqrt(np.sum(spectrum)) # 波高系数

# 生成海浪的二维空间分布
x = np.linspace(-10, 10, n)
y = np.linspace(-10, 10, n)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

for i in range(n):
    for j in range(n):
        Z += k * np.sqrt(spectrum_df.iloc[i, j]) * np.cos(2 * np.pi * (f1[i] * X + f2[j] * Y) + phase[i, j])

# 绘制等高线图

plt.figure(figsize=(10, 8))
plt.contourf(X, Y, Z, levels=20, cmap='viridis')
plt.colorbar()
plt.title('二维海面高度空间分布图（等高线图）')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('3.png')
plt.show()

# 绘制三维图

fig = plt.figure(figsize=(20, 16))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')

# 缩小z轴的比例
scale_x = 1
scale_y = 1
scale_z = 0.3

def short_proj():
    return np.dot(Axes3D.get_proj(ax), np.diag([scale_x, scale_y, scale_z, 1]))

ax.get_proj = short_proj

ax.set_title('三维海面高度空间分布图', y=0.9)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Z')
plt.savefig('4.png')
plt.show()


# 输出显示频谱并保存频谱DataFrame到Excel文件

print(spectrum_df)
spectrum_df.to_excel('spectrum.xlsx', index=False)

# 绘制频谱图
spectrum_df_diag = np.diagonal(spectrum_df.values)

plt.figure(figsize=(10, 8))
plt.title('频谱图')
plt.xlabel('频率')
plt.ylabel('能量')
plt.plot(spectrum_df_diag)
plt.savefig('5.png')
plt.show()