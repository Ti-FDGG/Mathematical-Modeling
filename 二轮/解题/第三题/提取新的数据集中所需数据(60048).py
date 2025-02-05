import numpy as np
import h5py
import pandas as pd
from scipy.signal import stft
with h5py.File(r"C:\Users\Timothy\Desktop\数学建模相关\二轮\数据集\20221112180016_stare_HH.mat", 'r') as data:
    acT12 = np.array(data['amplitude_complex_T1']) # 得到的数据类型是numpy.void
acT1_complex2 = np.array([complex(x[0], x[1]) for x in acT12.flat], 
                        dtype=complex).reshape(acT12.shape) # 选择所需要的部分，并将numpy.void转换为complex
np.save(r"C:\Users\Timothy\Desktop\数学建模相关\二轮\数据集\data_amplitude_T1_complex2.npy", 
        acT1_complex2) # 保存为.npy文件，以后读取时直接读取.npy文件即可，读取到的数据类型是complex