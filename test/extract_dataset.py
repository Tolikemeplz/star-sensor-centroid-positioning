import os
import json
import numpy as np
import pandas as pd


def filter_lines_by_keyword(source_file_path, output_dir, keyword):
    """
    从源文件中筛选包含关键字的行，并将这些行写入到输出目录的新文件中。
    新文件的文件名由keyword和val组成。

    :param source_file_path: 源文件的路径
    :param output_dir: 输出文件的目录路径
    :param keyword: 要搜索的关键字
    :param val: 文件名的一部分，用于生成输出文件的名称
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 创建输出文件的完整路径
    output_file_name = f"{keyword}_val.txt"
    output_file_path = os.path.join(output_dir, output_file_name)

    # 打开源文件
    with open(source_file_path, 'r') as source_file:
        # 读取所有行
        lines = source_file.readlines()

    # 假设 val_lines 已经通过前面的代码被赋值
    number_of_elements = len(lines)
    print(f"整个验证集 中包含 {number_of_elements} 张图片。")

    filter_num=0
    # 打开目标文件准备写入
    with open(output_file_path, 'w') as output_file:
        # 遍历所有行，检查是否包含关键字
        for line in lines:
            if keyword in line:
                output_file.write(line)
                filter_num += 1
    print(f'从验证集中筛选出{filter_num}张图片')

    return output_file_path

def apart_lines_by_factor(source_file_path,output_dir,csv_path,factor,apart):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 读取CSV文件
    df_csv = pd.read_csv(csv_path, low_memory=False)
    with open(source_file_path) as f:
        lines   = f.readlines()
    fileids_txt=[]
    fileids_withline = {}
    for annotation_line in lines:
        line = annotation_line.split()
        image_path = os.path.basename(line[0])
        fileids_txt.append(image_path)
        fileids_withline[f"{image_path}"] = annotation_line

    # 过滤出csv文件中存在于txt文件中的fileid
    df_csv = df_csv[df_csv['name'].isin(fileids_txt)]
    if factor == 'wxy':
        # 将'xtyt_p'列中的字符串转换为字典
        df_csv['xtyt_p'] = df_csv['xtyt_p'].apply(eval)
        # 提取'wx'键的值，并创建一个新列
        df_csv['wx'] = df_csv['xtyt_p'].apply(lambda x: x['wx'])
        # 提取'wx'键的值，并创建一个新列
        df_csv['wy'] = df_csv['xtyt_p'].apply(lambda x: x['wy'])
        # 计算wx和wy列的平方和的平方根
        df_csv['wxy'] = np.sqrt(df_csv['wx'] ** 2 + df_csv['wy'] ** 2)
        # 按照factor值排序
        df_sorted = df_csv.sort_values(by='wxy')
        # 获取factor值的最小值和最大值
        min_factor = df_sorted['wxy'].min()
        max_factor = df_sorted['wxy'].max()
    elif factor == 'wz':
        df_csv['xtyt_p'] = df_csv['xtyt_p'].apply(eval)
        df_csv['wz'] = df_csv['xtyt_p'].apply(lambda x: abs(x['wz']))
        df_sorted = df_csv.sort_values(by='wz')
        min_factor = df_sorted[f'{factor}'].min()
        max_factor = df_sorted[f'{factor}'].max()
    elif factor == 'gauss_mean':
        df_csv['gauss_p'] = df_csv['gauss_p'].apply(eval)
        df_csv['gauss_mean'] = df_csv['gauss_p'].apply(lambda x: x['mean'])
        df_sorted = df_csv.sort_values(by='gauss_mean')
        min_factor = df_sorted[f'{factor}'].min()
        max_factor = df_sorted[f'{factor}'].max()
    elif factor == 'gauss_variance':
        df_csv['gauss_p'] = df_csv['gauss_p'].apply(eval)
        df_csv['gauss_variance'] = df_csv['gauss_p'].apply(lambda x: x['variance'])
        df_sorted = df_csv.sort_values(by='gauss_variance')
        min_factor = df_sorted[f'{factor}'].min()
        max_factor = df_sorted[f'{factor}'].max()
    elif factor == 'dark_noise':
        df_csv['poisson_p'] = df_csv['poisson_p'].apply(eval)
        df_csv['dark_noise'] = df_csv['poisson_p'].apply(lambda x: x['dark_noise'])
        df_sorted = df_csv.sort_values(by='dark_noise')
        min_factor = df_sorted[f'{factor}'].min()
        max_factor = df_sorted[f'{factor}'].max()
    elif factor == 'e':
        df_csv['poisson_p'] = df_csv['poisson_p'].apply(eval)
        df_csv['e'] = df_csv['poisson_p'].apply(lambda x: x['e'])
        df_sorted = df_csv.sort_values(by='e')
        min_factor = df_sorted[f'{factor}'].min()
        max_factor = df_sorted[f'{factor}'].max()
    elif factor == 'mv':
        # 将 'mv' 列的值从字符串转换为数值
        df_csv['mv'] = pd.to_numeric(df_csv['mv'], errors='coerce')
        df_sorted = df_csv.sort_values(by=factor)
        min_factor = df_sorted[f'{factor}'].min()
        max_factor = df_sorted[f'{factor}'].max()

    # 计算每个组的factor值范围
    factor_range = (max_factor - min_factor) / apart

    # 存储每个组的中点值
    group_midpoints = []
    group_paths = []

    # 分组并写入新的txt文件
    for i in range(apart):
        # 计算当前组的factor值范围
        group_min = min_factor + i * factor_range
        group_max = min_factor + (i + 1) * factor_range if i != apart - 1 else max_factor
        group_midpoint = (group_min + group_max) / 2  # 计算中点值

        # 筛选出属于当前组的fileid
        group_df = df_sorted[(df_sorted[factor] >= group_min) & (df_sorted[factor] < group_max)]
        group_image_paths = group_df['name'].tolist()

        # 写入txt文件，文件名使用中点值
        output_file_name = f'{group_midpoint:.2f}.txt'  # 格式化中点值，保留两位小数
        output_file_path = os.path.join(output_dir, output_file_name)
        group_paths.append(output_file_path)
        # 写入txt文件
        with open(output_file_path, 'w') as f:
            for group_image_path in group_image_paths:
                f.write(fileids_withline[f"{group_image_path}"])
        # 存储每个组的中点值
        group_midpoints.append(group_midpoint)

    print('文件拆分完成。')
    return group_midpoints, group_paths  # 返回每个组的中点值数组

