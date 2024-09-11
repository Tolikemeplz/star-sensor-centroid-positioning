from PIL import Image
import numpy as np
import os
from tqdm import tqdm

# 切换到包含PNG图片的文件夹
os.chdir(r'D:\programing\课题代码-旧\official\dataset\pre\image-128')  # 请替换为您的文件夹路径

# 列出文件夹中所有的PNG文件
png_files = [f for f in os.listdir('.') if f.endswith('.png')]

# 初始化总和和总和的平方，用于计算均值和方差
total_sum = 0.0
total_sum_squared = 0.0
max_value = float('-inf')  # 初始化为负无穷大
min_value = float('inf')   # 初始化为正无穷大
total_count = 0

# 使用tqdm显示进度
for file in tqdm(png_files, desc='Processing images'):
    # 打开图片并转换为32位整数灰度
    img = Image.open(file).convert('I')
    # 将图片转换为NumPy数组
    img_np = np.array(img)

    # 更新最大值和最小值
    max_value = max(max_value, np.max(img_np))
    min_value = min(min_value, np.min(img_np))

    # 计算当前图片的像素总和和总和的平方
    sum_img = np.sum(img_np)
    sum_squared_img = np.sum(img_np ** 2)

    # 累加到总和中
    total_sum += sum_img
    total_sum_squared += sum_squared_img
    total_count += img_np.size

# 计算均值
mean = total_sum / total_count

# 计算方差
variance = (total_sum_squared / total_count) - mean ** 2

# 计算标准差
std_dev = np.sqrt(variance)

print(f"均值: {mean}")
print(f"标准差: {std_dev}")
print(f"最大值: {max_value}")
print(f"最小值: {min_value}")
