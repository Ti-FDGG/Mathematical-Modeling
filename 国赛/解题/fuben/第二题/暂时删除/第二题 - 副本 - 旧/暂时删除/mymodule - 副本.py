# 注意这里忽略了两个无法从源解析导入的警告
import numpy as np # type: ignore
import pandas as pd # type: ignore
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt # type: ignore
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

df11 = pd.read_excel('../excels/1.xlsx', sheet_name='乡村的现有耕地')
# print(df11)
df12 = pd.read_excel('../excels/1.xlsx', sheet_name='乡村种植的农作物')
df12.fillna(method='ffill', inplace=True)
# print(df12)
df21 = pd.read_excel('../excels/2.xlsx', sheet_name='2023年的农作物种植情况')
df21.fillna(method='ffill', inplace=True)
# print(df21)
df22 = pd.read_excel('../excels/2.xlsx', sheet_name='2023年统计的相关数据')
# 计算销售单价平均值
def calculate_average_price(price_str):
    prices = list(map(float, price_str.split('-')))
    return sum(prices) / len(prices)
df22['销售单价平均值/(元/斤)'] = df22['销售单价/(元/斤)'].apply(calculate_average_price)
# print(df22)

class Field:
    def __init__(self, field_name, field_type, field_area, crop_and_season, season, seasonmonth, planted_crop):
        self.field_name = field_name
        self.field_type = field_type
        self.field_area = field_area
        self.crop_and_season = crop_and_season
        self.season = season
        self.seasonmonth = seasonmonth
        self.planted_crop = planted_crop

    def __str__(self):
        return f'Field(planted_crop={self.planted_crop}, field_name={self.field_name}, field_type={self.field_type}, field_area={self.field_area}, crop_and_season={self.crop_and_season}, season={self.season}, seasonmonth={self.seasonmonth})'

class Crop:
    def __init__(self, crop_id, crop_name, crop_type, planting_fields, crop_price):
        self.crop_id = crop_id
        self.crop_name = crop_name
        self.crop_type = crop_type
        self.planting_fields = planting_fields
        self.crop_price = crop_price

    def __repr__(self):
        return f"Crop(crop_id={self.crop_id}, crop_name={self.crop_name}, crop_type={self.crop_type}, planting_fields={self.planting_fields}, crop_price={self.crop_price})"

def create_fields(data):
    fields = []
    for index, row in data.iterrows():
        field_name = row['地块名称']
        field_type = row['地块类型']

        if field_type[-1] == ' ':
            # 很奇怪，普通大棚后面有个空格
            field_type = field_type.rstrip(' ')

        field_area = row['地块面积/亩']
        planted_crop = []
        # 处理种植作物和季节字符串
        crop_and_season_str = row['种植作物和季节'].rstrip('0')
        crop_and_season_groups = crop_and_season_str.split(',')
        crop_and_season = [group.split() for group in crop_and_season_groups]    
        # 初始化 seasonmonth 变量
        seasonmonth = []
        season = 0
        seasonmonth_str = row['季节的月份']
        if seasonmonth_str != 0:
            seasonmonth_groups = seasonmonth_str.split(',')
            seasonmonth = [group.split() for group in seasonmonth_groups]
        field = Field(field_name, field_type, field_area, crop_and_season, season, seasonmonth, planted_crop)
        fields.append(field)
    return fields

def create_new_fields(fields):
    new_fields = []
    for field in fields:
        if field.field_type in ['平旱地', '梯田', '山坡地']:
            field.season = '单季'
            new_fields.append(field)
        elif field.field_type == '水浇地':
            field_single = Field(field.field_name, field.field_type, field.field_area, field.crop_and_season, '单季', field.seasonmonth, field.planted_crop)
            field_first = Field(field.field_name, field.field_type, field.field_area, field.crop_and_season, '第一季', field.seasonmonth, field.planted_crop)
            field_second = Field(field.field_name, field.field_type, field.field_area, field.crop_and_season, '第二季', field.seasonmonth, field.planted_crop)
            new_fields.extend([field_single, field_first, field_second])
        elif field.field_type in ['普通大棚', '智慧大棚']:
            field_first = Field(field.field_name, field.field_type, field.field_area, field.crop_and_season, '第一季', field.seasonmonth, field.planted_crop)
            field_second = Field(field.field_name, field.field_type, field.field_area, field.crop_and_season, '第二季', field.seasonmonth, field.planted_crop)
            new_fields.extend([field_first, field_second])
    return new_fields

def create_crops(df12, df22):
    # 创建作物价格映射
    crop_price_map = df22.set_index('作物编号')['销售单价平均值/(元/斤)'].to_dict()
    
    crops = []
    for index, row in df12.iterrows():
        crop_id = row['作物编号']
        crop_name = row['作物名称']
        if crop_name[-1] == ' ':
            # 去除作物名称末尾的空格
            crop_name = crop_name.rstrip(' ')
        
        crop_type = row['作物类型']
        crop_price = crop_price_map.get(crop_id, None)  # 从映射中获取销售单价平均值
        
        # 处理种植耕地字符串
        planting_fields_str = row['种植耕地'].rstrip('0')
        planting_fields_groups = planting_fields_str.split(',')
        planting_fields = [group.split() for group in planting_fields_groups]
        
        # 初始化 Crop 对象并添加到列表中
        crop = Crop(crop_id, crop_name, crop_type, planting_fields, crop_price)
        crops.append(crop)
    
    # 创建亩产量和种植成本映射
    yield_map = df22.set_index(['作物编号', '地块类型', '种植季次'])['亩产量/斤'].to_dict()
    cost_map = df22.set_index(['作物编号', '地块类型', '种植季次'])['种植成本/(元/亩)'].to_dict()
    
    # 遍历 crops 列表，进一步遍历 planting_fields 的每一个二项列表值，追加亩产量和种植成本
    for crop in crops:
        for field in crop.planting_fields:
            field_type = field[0]
            season = field[1]
            key = (crop.crop_id, field_type, season)
            
            crop_yield = yield_map.get(key, None)
            crop_cost = cost_map.get(key, None)
            
            # 当且仅当作物编号、地块类型和种植季节都相同时，追加数据
            if crop_yield is not None and crop_cost is not None:
                field.append(crop_yield)
                field.append(crop_cost)
    
    return crops

def get_yield(field, crop):
    # 获取 field_type 和 season
    field_type = field.field_type
    season = field.season

    # 在 crop 的 planting_fields 属性中进行匹配
    for pf in crop.planting_fields:
        if pf[0] == field_type and pf[1] == season:
            return pf[2]  # 返回第三个元素的值，即亩产量

    return 0  # 如果未找到合适的项，返回 0

def get_cost(field, crop):
    # 获取 field_type 和 season
    field_type = field.field_type
    season = field.season

    # 在 crop 的 planting_fields 属性中进行匹配
    for pf in crop.planting_fields:
        if pf[0] == field_type and pf[1] == season:
            return pf[3]  # 返回第四个元素的值，即种植成本

    return 0  # 如果未找到合适的项，返回 0

def get_expected_sales(crop, expected_sales):
    # 在 expected_sales 中找到对应的行
    row = expected_sales[expected_sales['作物名称'] == crop.crop_name]
    
    if not row.empty:
        # 返回总产量/斤列的值
        return row['总产量/斤'].values[0]
    
    return 0  # 如果未找到对应的作物，返回 0

def check_crop_constraints(field, crop):
    for pf in crop.planting_fields:
        if pf[0] == field.field_type and pf[1] == field.season:
            return True
    return False

def check_rotation_constraints(field, crop, df_last_result):
    # 获取地块名称、作物名称和季次
    field_name = field.field_name
    crop_name = crop.crop_name
    season = field.season
    if season == '单季':
        season = '第一季'

    # 检查 df2024_result 中对应地块、季次和作物的种植面积
    for index, row in df_last_result.iterrows():
        if row['种植地块'] == field_name and row['种植季次'] == season and row[crop_name] > 0:
            return False  # 如果种植面积不为0，返回 False

    return True  # 如果种植面积为0，返回 True

def update_new_fields(variables, new_fields):
    for v in variables.values():
        # 解析 v.name 得到 field_name, crop_name 和 season
        field_name, crop_name, season = v.name.split('_')
        # 找到对应的 Field 对象
        for field in new_fields:
            if field.field_name == field_name and field.season == season:
                # 向 planted_crop 元素追加一个二元列表
                field.planted_crop.append([crop_name, v.varValue])
                break

def output(df_template, year, new_fields, k):
    # 将读取到的表格中的NaN取值为0
    df_template.fillna(0, inplace=True)
    df_template.columns = [column.rstrip(' ') for column in df_template.columns]
    for field in new_fields:
        season = "第一季" if field.season == "单季" else field.season
        field_name = field.field_name

        row_index = df_template[(df_template.iloc[:, 0] == season) & (df_template.iloc[:, 1] == field_name)].index

        if not row_index.empty:
            row_index = row_index[0]
            for crop in field.planted_crop:
                crop_name = crop[0]
                crop_area = crop[1]
                # 根据 crop_area 的值调整输出格式
                if crop_area > 1:
                    crop_area = int(crop_area)
                else:
                    crop_area = round(crop_area, 1)
                # 找到对应的列
                col_index = df_template.columns.get_loc(crop_name)
                # 更新单元格
                df_template.iloc[row_index, col_index] = crop_area

    # # 在最后一列添加每一行所有种植面积的求和
    # df_template['总种植面积'] = df_template.iloc[:, 2:].sum(axis=1)

    df_template.to_excel(f'../excels/附件3/{year}_result_{k}.xlsx', index=False)

def get_randoms(std_devs):
    """
    根据给定的标准差列表生成随机数列表。

    参数:
    std_devs (list): 包含每个随机数标准差的列表

    返回:
    list: 包含生成的随机数的列表，按以下顺序排列：
        - 小麦和玉米的预期销售量增长率，介于0.05~0.1之间
        - 其他农作物预期销售量相对于2023年的变化率，介于-0.05~+0.05之间
        - 农作物亩产量每年的变化率，介于-0.1~+0.1之间
        - 农作物种植成本每年的增长率，在0.05附近
        - 粮食类作物销售价格每年的增长率，在0附近
        - 蔬菜类作物销售价格每年的增长率，在0.05附近
        - 食用菌的销售价格每年的增长率，在-0.05~-0.01之间
        - 羊肚菌的销售价格每年下降幅度为5%
    """
    # 定义均值
    means = [
        (0.05 + 0.1) / 2,  # es_wheetandcorn_growth_mean
        (-0.05 + 0.05) / 2,  # es_others_rate_mean
        (-0.1 + 0.1) / 2,  # ty_rate_mean
        0.05,  # pc_growth_mean
        0,  # sp_growth_mean
        0.05,  # vp_growth_mean
        (-0.05 + -0.01) / 2  # mp_growth_mean
    ]
    
    # 生成随机数
    randoms = [np.random.normal(loc=mean, scale=std) for mean, std in zip(means, std_devs)]
    
    # 添加固定值
    randoms.append(-0.05)  # m1p_growth
    
    return randoms
