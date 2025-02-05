import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 获取当前文件的完整路径
current_file_path = __file__

# 获取当前文件所在目录的路径
current_dir = os.path.dirname(current_file_path)

# 改变当前工作目录
os.chdir(current_dir)
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False
# 读取当前目录下的222.xlsx文件中的"英国"和"日本"这两个Sheet的数据
data_uk = pd.read_excel('数据/222.xlsx', sheet_name='英国1993').T
data_uk.columns = ['GDP', 'GDP增长率', '债务']
data_japan = pd.read_excel('数据/222.xlsx', sheet_name='日本1993').T
data_japan.columns = ['GDP', 'GDP增长率', '债务']
# print(data_uk.head())
df = pd.read_excel('数据/GDP、GDP增长率、美债.xlsx')
# 定义转换函数
def quarter_to_date(quarter_str):
    year, quarter = quarter_str.split('年')
    quarter_dict = {"第一季度": "03-31", "第二季度": "06-30", "第三季度": "09-30", "第四季度": "12-31"}
    return pd.to_datetime(f"{year}-{quarter_dict[quarter]}")
columns = df.columns
new_columns = [quarter_to_date(col) for col in columns]
df.columns = new_columns
df = df.T
df.columns = ['GDP增长率', 'GDP', '美债']
# print(df.head())
# 将df重采样到年
data_us = df.resample('YE').mean().dropna()
# print(df_yearly.head())

countries_data = {
    '英国': data_uk,
    '日本': data_japan,
    '美国': data_us
}

fig = plt.figure(figsize=(18, 6))

for i, (country, data) in enumerate(countries_data.items(), start=1):
    ax = fig.add_subplot(1, 3, i, projection='3d')
    ax.scatter(data['GDP'], data['GDP增长率'], data['美债'] if country == '美国' else data['债务'])
    ax.set_title(country)
    ax.set_xlabel('GDP')
    ax.set_ylabel('GDP增长率')
    ax.set_zlabel('债务')

plt.tight_layout()
plt.show()