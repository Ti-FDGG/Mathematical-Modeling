import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

# 定义JONSWAP函数
def JONSWAP(f, f_p, alpha, gamma):
    sigma = np.zeros_like(f)
    sigma[f <= f_p] = 0.07
    sigma[f > f_p] = 0.09
    Sw = alpha / f**5 * np.exp(-1.25 * (f_p / f)**4) * gamma**np.exp(- (f - f_p)**2 / (2 * sigma**2 * f_p**2))
    return Sw

# 定义蒙特卡洛模拟
def monte_carlo_simulation_1d(n, f_p, alpha, gamma):
    f = np.linspace(0.01, 2, n)  # 频率范围
    spectrum = JONSWAP(f, f_p, alpha, gamma)  # 计算频谱
    phase = np.random.uniform(0, 2*np.pi, n)  # 随机相位
    return f, spectrum, phase

# 参数设置
n = 200  # 模拟的频率点数
f_p = 0.1
alpha = 0.0081
gamma = 3.3
hmax = 2.8
kfix = 0.6 # 修正系数
# 进行蒙特卡洛模拟
f, spectrum, phase = monte_carlo_simulation_1d(n, f_p, alpha, gamma)
k = kfix * hmax/np.sqrt(sum(spectrum)) # 波高系数

# 生成一维海浪的空间分布
x = np.linspace(-10, 10, n)
Z = np.zeros_like(x)

fig = plt.figure(figsize=(20, 16))
ax = fig.add_subplot(111, projection='3d')

# 在三维空间中的不同位置绘制1D海浪的空间分布的正弦展开

nn = 25 # 展示的展开项数

for i in range(n):
    Z += k * np.sqrt(spectrum[i]) * np.cos(2 * np.pi * f[i] * x + phase[i])
for i in range(1, nn):
    y = k * np.sqrt(spectrum[i]) * np.cos(2 * np.pi * f[i] * x + phase[i])
    ax.plot(x, y, zs=i, zdir='y', alpha=1-(1/nn)*i)
ax.plot(x, Z, zs=0, zdir='y', linewidth=2, color='black')
# ax.grid(False)
# 改变坐标轴的比例
scale_x = 1
scale_y = 1.8
scale_z = 0.8

def short_proj():
    return np.dot(Axes3D.get_proj(ax), np.diag([scale_x, scale_y, scale_z, 1]))
ax.get_proj = short_proj
ax.set_zlim(-3, 3)
plt.title('一维海面高度空间分布图（正弦展开）', y=1.2)
plt.savefig('1.png')
plt.show()


# 绘制1D海浪的平面图
plt.figure(figsize=(20, 16))
plt.plot(x, Z)
plt.xlabel('x')
plt.ylabel('Z')
plt.ylim(-3, 3)
plt.title('一维海面高度空间分布图')
plt.savefig('2.png')
plt.show()