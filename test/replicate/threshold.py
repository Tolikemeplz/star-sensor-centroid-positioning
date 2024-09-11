import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import mean

def imply_threshold(image, border_width=10, m=3):
    """
    计算图像的阈值。

    参数:
    - image: 输入的灰度图像。
    - border_width: 图像外围边框的宽度，默认为10。

    返回:
    - 计算得到的阈值。
    """

    # 分别获取四个边框的像素，并将它们展平成一维数组
    top_border = image[0:border_width, :].flatten()
    bottom_border = image[-border_width:, :].flatten()
    left_border = image[:, 0:border_width].flatten()
    right_border = image[:, -border_width:].flatten()

    # 合并这些边框像素
    border_pixels = np.concatenate([top_border, bottom_border, left_border, right_border])

    # 计算边框像素的平均灰度值
    border_mean = np.mean(border_pixels)

    # 计算整张图像的均方根
    rms = np.sqrt(np.mean(image**2))

    # 设置阈值
    threshold = border_mean  + m * rms

    segmented_image = image > threshold

    return segmented_image

def traditionel_threshold(image, border_width=30, m=3):
    # 分别获取四个边框的像素，并将它们展平成一维数组
    top_border = image[0:border_width, :].flatten()
    bottom_border = image[-border_width:, :].flatten()
    left_border = image[:, 0:border_width].flatten()
    right_border = image[:, -border_width:].flatten()

    # 合并这些边框像素
    border_pixels = np.concatenate([top_border, bottom_border, left_border, right_border])

    # 计算边框像素的平均灰度值
    border_mean = np.mean(border_pixels)

    # 计算整张图像的均方根
    rms = np.sqrt(np.mean(image ** 2))

    # 设置阈值
    threshold = border_mean + m * rms

    return threshold

def triangle_threshold(image):
    # 计算图像的灰度直方图
    hist, _ = np.histogram(image, bins=range(1024))

    # 归一化直方图
    hist = hist / hist.sum()

    # 初始化
    max_area = 0
    best_threshold = 0

    # 遍历所有可能的阈值
    for threshold in range(1, 1023):
        # 计算左边的总和和权重总和
        left_sum = hist[:threshold].sum()
        left_weight_sum = np.dot(np.arange(threshold), hist[:threshold])

        # 计算右边的总和和权重总和
        right_sum = hist[threshold:].sum()
        right_weight_sum = np.dot(np.arange(threshold, 1023), hist[threshold:])

        # 如果左边或右边为空，则跳过
        if left_sum == 0 or right_sum == 0:
            continue

            # 计算三角形的面积
        area = abs(left_weight_sum / left_sum - right_weight_sum / right_sum)

        # 更新最大面积和最佳阈值
        if area > max_area:
            max_area = area
            best_threshold = threshold

            # 应用阈值

    return best_threshold

def imply_triangle_threshold(image):
    thres = triangle_threshold(image)
    segmented_image = image > thres
    return segmented_image


if __name__ == '__main__':
  image_path = 'point_both_13925.png'  # 替换为你的图片路径
  image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
  segmented_image = imply_threshold(image,m=3)
  # 使用matplotlib显示分割后的图像
  plt.imshow(segmented_image, cmap='gray')
  plt.title('Segmented Image')
  plt.axis('off')  # 关闭坐标轴
  plt.show()

