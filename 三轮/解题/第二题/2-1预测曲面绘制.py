import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os
from sklearn.preprocessing import PolynomialFeatures
# 获取当前文件的完整路径
current_file_path = __file__

# 获取当前文件所在目录的路径
current_dir = os.path.dirname(current_file_path)

# 改变当前工作目录
os.chdir(current_dir)
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_excel('GDP、GDP增长率、美债.xlsx')
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
df_1 = df[df.index <= '2008-09-30']
df_2 = df[(df.index >= '2009-03-31') & (df.index <= '2019-12-31')]
df_3 = df[df.index >= '2020-12-31']
dfs = [df_1, df_2, df_3]
# 梯度下降法更新系数
def gradient_descent(X, y, theta, learning_rate=0.001):
    iterations=10000 * (100//y.shape[0])
    m = len(y)
    cost_history = np.zeros(iterations)
    for it in range(iterations):
        prediction = np.dot(X, theta)
        theta = theta - (1/m) * learning_rate * (X.T.dot((prediction - y)))
        cost_history[it] = (1/(2*m)) * np.sum(np.square(prediction - y))
    return theta, cost_history

# 在应用梯度下降之前对特征进行缩放
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

r_squared_values = np.load('r_squared_values.npy')

for i, dfi in enumerate(dfs):
    X = dfi[['GDP', 'GDP增长率']]
    y = dfi['美债'].values
    X_scaled = scaler.fit_transform(X) # 输入的是标准化后的值
    
    # 找到R方值最大的组合
    max_r_squared_index = np.unravel_index(np.argmax(r_squared_values[:, :, i]), r_squared_values[:, :, i].shape)
    degree_gdp_best, degree_growth_best = max_r_squared_index
    
    # 重新构建模型
    poly_gdp_best = PolynomialFeatures(degree=degree_gdp_best+1, include_bias=False)
    poly_growth_best = PolynomialFeatures(degree=degree_growth_best+1, include_bias=False)
    X_gdp_poly_best = poly_gdp_best.fit_transform(X_scaled[:, 0].reshape(-1, 1))
    X_growth_poly_best = poly_growth_best.fit_transform(X_scaled[:, 1].reshape(-1, 1))
    
    X_poly_best = np.concatenate((X_gdp_poly_best, X_growth_poly_best), axis=1)
    X_poly_best = sm.add_constant(X_poly_best)  # 添加常数项
    # 初始化系数
    theta_best = np.random.randn(X_poly_best.shape[1], 1)
    
    # 使用梯度下降法求解系数
    theta_best, _ = gradient_descent(X_poly_best, y.reshape(-1,1), theta_best)
    
    # 使用求解的系数计算预测值
    predictions = X_poly_best.dot(theta_best)
    
    # 计算残差
    residuals = y.reshape(-1,1) - predictions
    # 在原有代码基础上增加以下部分

    # 为了生成函数表达式，我们需要特征的名称
    feature_names_gdp = poly_gdp_best.get_feature_names_out(['g'])
    feature_names_growth = poly_growth_best.get_feature_names_out(['ggr'])
    feature_names = np.concatenate((feature_names_gdp, feature_names_growth))

    # 构建函数表达式
    expression = f"美债预测值 = {theta_best.flatten()[0]:.3f}"
    for coef, name in zip(theta_best.flatten()[1:], feature_names):
        if coef >= 0:
            expression += f"+ {coef:.3f}*{name} "
        else:
            expression += f"- {-coef:.3f}*{name} "
    print(f"数据集 {i+1} 的函数表达式:\n{expression}\n")
    

# 创建一个新的图形
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')  # 使用111表示在一个1x1网格的第一个位置创建子图

for i, dfi in enumerate(dfs):
    X = dfi[['GDP', 'GDP增长率']]
    y = dfi['美债'].values
    X_scaled = scaler.fit_transform(X)  # 输入的是标准化后的值

    # 找到R方值最大的组合
    max_r_squared_index = np.unravel_index(np.argmax(r_squared_values[:, :, i]), r_squared_values[:, :, i].shape)
    degree_gdp_best, degree_growth_best = max_r_squared_index

    # 重新构建模型
    poly_gdp_best = PolynomialFeatures(degree=degree_gdp_best+1, include_bias=False)
    poly_growth_best = PolynomialFeatures(degree=degree_growth_best+1, include_bias=False)
    X_gdp_poly_best = poly_gdp_best.fit_transform(X_scaled[:, 0].reshape(-1, 1))
    X_growth_poly_best = poly_growth_best.fit_transform(X_scaled[:, 1].reshape(-1, 1))

    X_poly_best = np.concatenate((X_gdp_poly_best, X_growth_poly_best), axis=1)
    X_poly_best = sm.add_constant(X_poly_best)  # 添加常数项
    # 初始化系数
    theta_best = np.random.randn(X_poly_best.shape[1], 1)

    # 使用梯度下降法求解系数
    theta_best, _ = gradient_descent(X_poly_best, y.reshape(-1,1), theta_best)

    # 创建网格数据
    gdp_range = np.linspace(X_scaled[:, 0].min(), X_scaled[:, 0].max(), 20)
    growth_range = np.linspace(X_scaled[:, 1].min(), X_scaled[:, 1].max(), 20)
    gdp_grid, growth_grid = np.meshgrid(gdp_range, growth_range)

    # 计算网格上每一点的预测值
    X_grid_poly_gdp = poly_gdp_best.transform(gdp_grid.reshape(-1, 1))
    X_grid_poly_growth = poly_growth_best.transform(growth_grid.reshape(-1, 1))
    X_grid_poly = np.concatenate((X_grid_poly_gdp, X_grid_poly_growth), axis=1)
    X_grid_poly = sm.add_constant(X_grid_poly)  # 添加常数项
    predictions_grid = X_grid_poly.dot(theta_best)

    # 将gdp_grid和growth_grid转换回原始尺度
    gdp_growth_grid_scaled = np.vstack((gdp_grid.flatten(), growth_grid.flatten())).T
    gdp_growth_grid_original = scaler.inverse_transform(gdp_growth_grid_scaled)
    gdp_grid_original, growth_grid_original = gdp_growth_grid_original[:, 0].reshape(gdp_grid.shape), gdp_growth_grid_original[:, 1].reshape(growth_grid.shape)

    # 绘制预测曲面
    ax.plot_surface(gdp_grid_original, growth_grid_original, predictions_grid.reshape(gdp_grid.shape), alpha=0.3)

    # 绘制原始数据的三维散点图
    X_original = scaler.inverse_transform(X_scaled)
    ax.scatter(X_original[:, 0], X_original[:, 1], y, color='blue')

ax.set_xlabel('GDP（单位：十万亿美元）')
ax.set_ylabel('GDP增长率（百分比）')
ax.set_zlabel('美债（单位：十万亿美元）')
ax.set_title('原始数据三维散点图和 BGD 预测曲面')
plt.show()