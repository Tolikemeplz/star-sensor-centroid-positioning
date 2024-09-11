from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

pic_path = r'D:\！！工作学习\理化所\！！课题\高动态星敏\used_file\temp\1_const_linear_None_0.png'

image = Image.open(pic_path).convert('I')

# 将图片转换为numpy数组
image_array = np.array(image)

# 使用matplotlib显示图片，使用'viridis'颜色映射
plt.imshow(image_array, cmap='inferno')
# 创建一个颜色条，设置最小刻度和最大刻度
# cbar = plt.colorbar(ticks=np.linspace(0, 1023, 5))  # 生成0到1023之间的5个刻度位置
# cbar.set_ticks(np.linspace(0, 1023, 5))  # 设置刻度位置
# cbar.ax.set_yticklabels(['0', '256', '512', '768', '1023'])  # 设置刻度标签
# 创建一个颜色条，设置最小刻度和最大刻度
# cbar = plt.colorbar()
#
# # 设置刻度位置
# tick_positions = np.linspace(image_array.min(), image_array.max(), 5)
# cbar.set_ticks(tick_positions)
#
# # 设置刻度标签
# tick_labels = [str(int(tick)) for tick in tick_positions]
# cbar.ax.set_yticklabels(tick_labels)


plt.show()
