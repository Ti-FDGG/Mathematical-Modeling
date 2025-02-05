# 注意这里忽略了两个无法从源解析导入的警告

import pulp # type: ignore
import pandas as pd # type: ignore
from mymodule import *

def create_variables(new_fields, crops):
    # 创建决策变量
    variables = {}
    binary_variables = {}
    for field in new_fields:
        for crop in crops:
            var_name = f"{field.field_name}_{crop.crop_name}_{field.season}"
            variables[(field.field_name, crop.crop_name, field.season)] = pulp.LpVariable(var_name, lowBound=0)
            binary_var_name = f"binary_{var_name}"
            binary_variables[(field.field_name, crop.crop_name, field.season)] = pulp.LpVariable(binary_var_name, cat='Binary')
    
    # 创建实际销售量变量
    actual_sales = {}
    for crop in crops:
        actual_sales[crop.crop_name] = pulp.LpVariable(f"actual_sales_{crop.crop_name}", lowBound=0)

    # 创建超额部分变量
    excess_yield = {}
    for crop in crops:
        excess_yield[crop.crop_name] = pulp.LpVariable(f"excess_yield_{crop.crop_name}", lowBound=0)
    
    return variables, binary_variables, actual_sales, excess_yield

# 定义目标函数的各个部分
def define_objective_function(variables, actual_sales, excess_yield, crops, new_fields, k):
    '''
    k: 超额部分处理系数

    返回: 
    目标函数 profit
    '''
    # 收入部分：实际销售量 * 作物价格
    revenue = pulp.lpSum([
        actual_sales[crop.crop_name] * crop.crop_price
        for crop in crops
    ])

    # 成本部分：种植成本
    cost = pulp.lpSum([
        variables[(field.field_name, crop.crop_name, field.season)] * get_cost(field, crop)
        for field in new_fields for crop in crops
    ])

    # 超额部分处理：超额部分 * 作物价格 * 系数
    excess_handling = pulp.lpSum([
        excess_yield[crop.crop_name] * crop.crop_price * k
        for crop in crops
    ])

    # 总利润 = 收入 - 成本 + 超额部分处理
    profit = revenue - cost + excess_handling

    return profit

def add_constraints(prob, variables, binary_variables, actual_sales, excess_yield, 
                    crops, new_fields, expected_sales_data, df_last1_result, df_last2_result,
                    min_area_percent, max_plots):
    '''
    expected_sales_data: \n
    作物预期销售量数据，第1问中即为total_yield_2023。注意后面提供expected_sales_data的时候要注意这个数据的格式。\n
    df_last1_result:\n
    上一年的种植结果，用于检查重茬限制和豆类作物种植要求。\n
    df_last2_result:\n
    上上一年的种植结果，用于检查豆类作物种植要求。\n
    min_area_percent:\n
    作物在单个地块某一季节种植面积不宜太小的最小百分比。\n
    max_plots:\n
    作物种植不宜太分散的最大地块数。\n
    '''
    
    # 添加约束条件
    # 地块面积限制
    for field in new_fields:
        prob += pulp.lpSum([variables[(field.field_name, crop.crop_name, field.season)] for crop in crops]) <= field.field_area

    # 作物种植限制
    for field in new_fields:
        for crop in crops:
            if not check_crop_constraints(field, crop):
                prob += variables[(field.field_name, crop.crop_name, field.season)] == 0

    # 重茬限制
    for field in new_fields:
        for crop in crops:
            if not check_rotation_constraints(field, crop, df_last1_result):
                prob += variables[(field.field_name, crop.crop_name, field.season)] == 0

    # 豆类作物种植要求
    for field in new_fields:
        # 约束：确保三年内至少有一年种植过豆类作物
        prob += (
            pulp.lpSum([
                binary_variables[(field.field_name, crop.crop_name, field.season)]
                for crop in crops if crop.crop_type in ['粮食（豆类）', '蔬菜（豆类）']
            ]) +
            (df_last1_result[(df_last1_result['种植地块'] == field.field_name) & 
                            (df_last1_result['种植季次'] == field.season)]
            .iloc[:, 2:]
            .apply(lambda row: any(row[crop_name] > 0 for crop_name in row.index if crop_name in [crop.crop_name for crop in crops if crop.crop_type in ['粮食（豆类）', '蔬菜（豆类）']]), axis=1)
            .any()) +
            (df_last2_result[(df_last2_result['种植地块'] == field.field_name) & 
                            (df_last2_result['种植季次'] == field.season)]
            .iloc[:, 2:]
            .apply(lambda row: any(row[crop_name] > 0 for crop_name in row.index if crop_name in [crop.crop_name for crop in crops if crop.crop_type in ['粮食（豆类）', '蔬菜（豆类）']]), axis=1)
            .any()) >= 1
        )
    
    # 实际销售量约束
    for crop in crops:
        actual_yield = pulp.lpSum([
            variables[(field.field_name, crop.crop_name, field.season)] * get_yield(field, crop)
            for field in new_fields
        ])
        expected_sales = get_expected_sales(crop, expected_sales_data)
        prob += actual_sales[crop.crop_name] <= actual_yield
        prob += actual_sales[crop.crop_name] <= expected_sales

        # 超额部分约束
        prob += excess_yield[crop.crop_name] == actual_yield - expected_sales # 这里不是>=，应该是==？！
        prob += excess_yield[crop.crop_name] >= 0

    # 作物种植不宜太分散
    for crop in crops:
        for field_type in set(field.field_type for field in new_fields):
            if field_type not in ['普通大棚', '智慧大棚']:
                prob += pulp.lpSum([
                    binary_variables[(field.field_name, crop.crop_name, field.season)]
                    for field in new_fields if field.field_type == field_type
                ]) <= max_plots
 
    return prob

def add_constraints2(prob, variables, binary_variables, actual_sales, excess_yield, 
                    crops, new_fields, expected_sales_data, df_last1_result,
                    min_area_percent, max_plots):
    '''
    专用于2024的添加限制条件函数。可不写进论文附录
    '''
    # 添加约束条件
    # 地块面积限制
    for field in new_fields:
        prob += pulp.lpSum([variables[(field.field_name, crop.crop_name, field.season)] for crop in crops]) <= field.field_area

    # 作物种植限制
    for field in new_fields:
        for crop in crops:
            if not check_crop_constraints(field, crop):
                prob += variables[(field.field_name, crop.crop_name, field.season)] == 0

    # 重茬限制
    for field in new_fields:
        for crop in crops:
            if not check_rotation_constraints(field, crop, df_last1_result):
                prob += variables[(field.field_name, crop.crop_name, field.season)] == 0

    # 实际销售量约束
    for crop in crops:
        actual_yield = pulp.lpSum([
            variables[(field.field_name, crop.crop_name, field.season)] * get_yield(field, crop)
            for field in new_fields
        ])
        expected_sales = get_expected_sales(crop, expected_sales_data)
        prob += actual_sales[crop.crop_name] <= actual_yield
        prob += actual_sales[crop.crop_name] <= expected_sales

        # 超额部分约束
        prob += excess_yield[crop.crop_name] == actual_yield - expected_sales # 这里不是>=，应该是==？！
        prob += excess_yield[crop.crop_name] >= 0

    # 作物种植不宜太分散
    max_plots = 4
    for crop in crops:
        for field_type in set(field.field_type for field in new_fields):
            if field_type not in ['普通大棚', '智慧大棚']:
                prob += pulp.lpSum([
                    binary_variables[(field.field_name, crop.crop_name, field.season)]
                    for field in new_fields if field.field_type == field_type
                ]) <= max_plots

    return prob

def optimize_planting_strategy(new_fields, crops, expected_sales, df_last1_result, df_last2_result, params_list):
    """
    优化种植策略

    参数:
    new_fields (list): 包含 Field 对象的列表
    crops (list): 包含 Crop 对象的列表
    expected_sales (DataFrame): 作物预期销售量数据
    df_last1_result (DataFrame): 上一年的种植结果
    df_last2_result (DataFrame): 上上一年的种植结果
    params_list (list): 参数列表，包含 [k, min_area_percent, max_plots, year]

    返回：
    None
    """
    # 解包参数列表
    k, min_area_percent, max_plots, year = params_list

    # 创建决策变量和二进制变量
    variables, binary_variables, actual_sales, excess_yield = create_variables(new_fields, crops)

    # 创建线性规划问题
    prob = pulp.LpProblem("Crop_Planting_Optimization", pulp.LpMaximize)

    # 定义目标函数
    profit = define_objective_function(variables, actual_sales, excess_yield, crops, new_fields, k)

    # 将目标函数添加到问题中
    prob += profit

    # 添加约束条件
    prob = add_constraints(prob, variables, binary_variables, actual_sales, 
                           excess_yield, crops, new_fields, expected_sales, df_last1_result, df_last2_result,
                           min_area_percent, max_plots)

    # 求解问题
    prob.solve()
    print("Status:", pulp.LpStatus[prob.status])

    # 更新 new_fields
    update_new_fields(variables, new_fields)

    # 输出结果
    df_template = pd.read_excel('../excels/附件3/template.xlsx')
    output(df_template, year, new_fields, k)