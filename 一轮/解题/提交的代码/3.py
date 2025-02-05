# %% [markdown]
# ## 调用

# %%
import numpy as np
import pandas as pd

# plt.rcParams["figure.figsize"] = [27.50, 12.50]

excel = pd.read_excel("B题数据.xlsx")

ana = pd.DataFrame(excel).iloc[:, 2:]
# print(ana)

# %% [markdown]
# ## 第三题
# 
# 3、国家标准的I-V分类主要考虑水质本身对人和环境的影响，请从居民、游客生活体验的角度出发，对各个监测站的水质重新分类，并给出分类的依据。  
# *（分类，分类依据制定）*

# %% [markdown]
# ### 熵权法计算原始数据打分到二级标准（实际是一级）的权重
# 
# 注：不包括水质类别相对应的数据列。一方面是制定打分标准时忘记了，另一方面是后来反思也认为数据给出的水质类别与其他列是相关的，不必算进新的打分标准中

# %%
stdana = pd.DataFrame()
weight = []
k = 1 / np.log(len(ana.index))
for i, col in enumerate(ana):
    stdana[i] = ((ana[col]-min(ana[col]))/max(ana[col]-min(ana[col])))
    stdana[i] /= sum(stdana[i])
    for ji, j in enumerate(stdana[i]):
        if j != 0:
            stdana.iloc[ji, i] *= np.log(j)
        else:
            stdana.iloc[ji, i] = 0
    weight.append(1-(-k*sum(stdana[i])))
for k in range(len(weight)):
    weight[k] /= sum(weight)
weight = {k:v for k,v in zip(ana.columns, weight)}
# print(weight)

# %% [markdown]
# ### 分数计算

# %%
weight1 = [weight['水温(℃)'], weight['浊度(NTU)'], weight['溶解氧(mg/L)']]
weight2 = [weight['高锰酸盐指数(mg/L)'], weight['电导率(μS/cm)'], weight['pH(无量纲)'], weight['氨氮(mg/L)']]
weight3 = [weight['氨氮(mg/L)'], weight['溶解氧(mg/L)'], weight['总磷(mg/L)'], weight['总氮(mg/L)']]
weight1 /= sum(weight1)
weight2 /= sum(weight2)
weight3 /= sum(weight3)
weightn = [weight1, weight2, weight3]
s = pd.DataFrame(columns=['舒适度', '安全性', '清洁度'])
for i in range(3):
    score = pd.read_excel('分.xlsx', sheet_name=i)
    s.iloc[:, i] = np.dot(weightn[i], score.iloc[0:, 1:])
weightupper = [0.2, 0.4 ,0.4]
s_last = pd.concat([excel['断面名称'], pd.Series(np.dot(s, weightupper))], axis=1)
s_last.columns = ['断面名称', '总分']
s_last_sorted = sorted(s_last.values, key=lambda x: x[1], reverse=True)
# print(excel['断面名称'])
# print(s_last_sorted)
stdana[i] = ((ana[col]-min(ana[col]))/max(ana[col]-min(ana[col])))
total_scores = np.array([item[1] for item in s_last_sorted])
standardized_scores = (total_scores - min(total_scores)) / (max(total_scores) - min(total_scores))
levels = 6 - np.digitize(standardized_scores, np.linspace(-0.00001, 1.00001, 6), right=True)
s_last_sorted_with_scores = [np.append(item, [score, level]) for item, score, level in zip(s_last_sorted, standardized_scores, levels)]
for item in s_last_sorted_with_scores:
    print(item)
print(s_last_sorted)


