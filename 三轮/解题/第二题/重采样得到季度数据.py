import pandas as pd
import os
# 获取当前文件的完整路径
current_file_path = __file__
# 获取当前文件所在目录的路径
current_dir = os.path.dirname(current_file_path)
# 改变当前工作目录
os.chdir(current_dir)
df = pd.read_excel("数据.xlsx")

# 将时间列转换为时间格式，并将其设置为索引
df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
df.set_index(df.iloc[:, 0], inplace=True)

# 重采样到季度末，保留每个季度最后一条记录的值
df_quar = df.resample("QE").last()
# 保存数据
df_quar.iloc[:, 1].to_excel("季度数据.xlsx")