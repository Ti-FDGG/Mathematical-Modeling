import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

f = np.load(r"C:\Users\Timothy\Desktop\数学建模相关\二轮\数据集\acT1_cplx_stft_f.npy")
t = np.load(r"C:\Users\Timothy\Desktop\数学建模相关\二轮\数据集\acT1_cplx_stft_t.npy")
Zxx = np.load(r"C:\Users\Timothy\Desktop\数学建模相关\二轮\数据集\acT1_cplx_stft_Zxx.npy")
acT1_complex = np.load(r"C:\Users\Timothy\Desktop\数学建模相关\二轮\数据集\data_amplitude_T1_complex.npy")
df = pd.DataFrame(acT1_complex)
Zxx100p_max_f = np.argmax(np.abs(Zxx[100:, :, :]), axis=1)


Bone = np.array(f[Zxx100p_max_f])

# 创建images
# 遍历数据
for i in range(850):
    # 创建一个新的图像
    fig = plt.figure(figsize=(10, 6))
    plt.scatter(t, Bone[i], s=1)
    plt.ylim(-0.3, 0.3)
    # 保存图像到特定位置，名称为图像序号
    plt.savefig(f'C:/Users/Timothy/Desktop/数学建模相关/二轮/数据集/images/{i}.png')
    
    plt.close(fig)
# 创建spectrums
# 遍历数据
for i in range(850):
    # 创建一个新的图像
    fig = plt.figure(figsize=(10, 6))
    plt.specgram(df.iloc[i+100], NFFT=1024, Fs=2, noverlap=500, cmap='jet')
    plt.ylim(-0.3, 0.3)
    # 保存图像到特定位置，名称为图像序号
    plt.savefig(f'C:/Users/Timothy/Desktop/数学建模相关/二轮/数据集/spectrums/{i}.png')
    
    plt.close(fig)