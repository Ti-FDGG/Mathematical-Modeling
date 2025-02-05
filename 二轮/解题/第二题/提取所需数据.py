import numpy as np
import h5py
import pandas as pd
from scipy.signal import stft
with h5py.File(r"C:\Users\Timothy\Desktop\数学建模相关\二轮\数据集\20221112150043_stare_HH.mat", 'r') as data:
    acT1 = np.array(data['amplitude_complex_T1']) # 得到的数据类型是numpy.void
acT1_complex = np.array([complex(x[0], x[1]) for x in acT1.flat], 
                        dtype=complex).reshape(acT1.shape) # 选择所需要的部分，并将numpy.void转换为complex
np.save(r"C:\Users\Timothy\Desktop\数学建模相关\二轮\数据集\data_amplitude_T1_complex.npy", 
        acT1_complex) # 保存为.npy文件，以后读取时直接读取.npy文件即可，读取到的数据类型是complex


# 保存STFT结果
df = pd.DataFrame(acT1_complex)
# 短时傅里叶变换（STFT）
f, t, Zxx = stft(df, return_onesided=False)
np.save(r"C:\Users\Timothy\Desktop\数学建模相关\二轮\数据集\acT1_cplx_stft_f.npy", f)
np.save(r"C:\Users\Timothy\Desktop\数学建模相关\二轮\数据集\acT1_cplx_stft_t.npy", t)
np.save(r"C:\Users\Timothy\Desktop\数学建模相关\二轮\数据集\acT1_cplx_stft_Zxx.npy", Zxx)