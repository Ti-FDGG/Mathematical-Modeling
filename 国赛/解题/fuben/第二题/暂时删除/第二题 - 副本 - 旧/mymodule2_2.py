import numpy as np # type: ignore
import pandas as pd # type: ignore

def update_expected_sales_data(expected_sales_data, expected_sales_data_2023, std_devs):
    """
    更新预期销售量。

    参数:
    expected_sales (DataFrame): 包含当前预期销售量的 DataFrame
    expected_sales_2023 (DataFrame): 包含2023年预期销售量的 DataFrame
    std_devs (list): 包含每个随机数标准差的列表

    返回:
    DataFrame: 更新后的预期销售量 DataFrame
    """
    # 定义增长率的均值
    wheat_growth_mean = (0.05 + 0.1) / 2
    corn_growth_mean = (0.05 + 0.1) / 2
    others_growth_mean = (-0.05 + 0.05) / 2

    # 遍历 expected_sales 中的每一行
    for index, row in expected_sales_data.iterrows():
        crop_name = row['作物名称']
        
        if crop_name == '小麦':
            growth_rate = np.random.normal(loc=wheat_growth_mean, scale=std_devs[0])
            expected_sales_data.at[index, '预期销售量/斤'] *= (1 + growth_rate)
        
        elif crop_name == '玉米':
            growth_rate = np.random.normal(loc=corn_growth_mean, scale=std_devs[1])
            expected_sales_data.at[index, '预期销售量/斤'] *= (1 + growth_rate)
        
        else:
            # 从 expected_sales_2023 中找到对应的行
            row_2023 = expected_sales_data_2023[expected_sales_data_2023['作物名称'] == crop_name]
            if not row_2023.empty:
                base_sales = row_2023.iloc[0]['预期销售量/斤']
                growth_rate = np.random.normal(loc=others_growth_mean, scale=std_devs[2])
                expected_sales_data.at[index, '预期销售量/斤'] = base_sales * (1 + growth_rate)
    
    return expected_sales_data


def update_crops(crops, std_devs):
    """
    更新亩产量、种植成本和作物价格。

    参数:
    crops (list): 包含 Crop 对象的列表
    std_devs (list): 包含每个随机数标准差的列表

    返回:
    list: 更新后的 Crop 对象列表
    """
    # 定义增长率的均值
    yield_growth_mean = (-0.1 + 0.1) / 2
    cost_growth_mean = 0.05
    price_growth_mean_vegetables = 0.05
    price_growth_mean_mushrooms = (0.01 + 0.05) / 2

    # 遍历 crops 列表
    for crop in crops:
        for field in crop.planting_fields:
            # 更新亩产量
            yield_growth_rate = np.random.normal(loc=yield_growth_mean, scale=std_devs[3])
            field[2] *= (1 + yield_growth_rate)
            
            # 更新种植成本
            cost_growth_rate = np.random.normal(loc=cost_growth_mean, scale=std_devs[4])
            field[3] *= (1 + cost_growth_rate)
        
        # 更新作物价格
        if crop.crop_type in ['粮食', '粮食（豆类）', '粮食（水稻）']:
            price_growth_rate = np.random.normal(loc=0, scale=std_devs[5])
            crop.crop_price *= (1 + price_growth_rate)
        elif crop.crop_type in ['蔬菜', '蔬菜（水二）', '蔬菜（豆类）']:
            price_growth_rate = np.random.normal(loc=price_growth_mean_vegetables, scale=std_devs[6])
            crop.crop_price *= (1 + price_growth_rate)
        elif crop.crop_type == '食用菌':
            if crop.crop_name == '羊肚菌':
                crop.crop_price *= 0.95  # 下降5%
            else:
                price_growth_rate = np.random.normal(loc=price_growth_mean_mushrooms, scale=std_devs[7])
                crop.crop_price *= (1 + price_growth_rate)
    
    return crops
