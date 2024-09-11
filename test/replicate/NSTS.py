import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import morphology
from skimage import filters
from scipy.ndimage import binary_opening, binary_dilation, binary_erosion
from scipy import ndimage
from skimage.morphology import white_tophat
from skimage.filters import threshold_otsu
from test.replicate.threshold import traditionel_threshold,triangle_threshold


def create_matrix_with_border(m, d):
  # 使用 numpy 创建一个 m x m 的零矩阵
  matrix = np.zeros((m, m), dtype=int)

  # 将边缘宽度为 d 的外围一圈置为 1
  matrix[:d, :] = 1
  matrix[-d:, :] = 1
  matrix[:, :d] = 1
  matrix[:, -d:] = 1

  return matrix


def nst_star_segmentation(image, bs_size, bmi_size, bmo_size, be_size):
  # 定义结构元素
  bs = np.ones((bs_size, bs_size))
  bmi = np.ones((bmi_size, bmi_size))
  bmo = np.ones((bmo_size, bmo_size))
  dbm = int(1/2*(bmo_size-bmi_size))
  #print('dbm:',dbm)
  bm = create_matrix_with_border(bmo_size,dbm)
  # bm = np.ones((8, 8))
  be = np.ones((be_size, be_size))

  bm = bm.astype(np.uint8)
  be = be.astype(np.uint8)
  bs = bs.astype(np.uint8)

  # 执行灰度开运算
  # k = ndimage.grey_dilation(ndimage.grey_erosion(image, structure=bs), structure=bs)
  k = cv2.morphologyEx(image, cv2.MORPH_OPEN, bs)

  # 执行形态学操作，使用 BMI 和 BE 结构元素进行膨胀和腐蚀操作

  n = ndimage.grey_erosion(ndimage.grey_dilation(image, structure=bm), structure=be)
  n = cv2.erode(cv2.dilate(image, bm, iterations=1), be, iterations=1)
  n1 = cv2.dilate(image, bm, iterations=1)
  n2 = cv2.erode(n1,be,iterations=1)

  # 改进的顶帽变换分割星点
  # 将开运算结果与新的腐蚀结果相减，得到分割后的星图
  # r = k - np.minimum(k, n)
  r = k.astype(int) - np.minimum(k.astype(int), n.astype(int))
  r = k - np.minimum(k, n)

  # 应用阈值切割
  # thresh = threshold_otsu(r)*2
  #thresh = triangle_threshold(r)
  thresh = traditionel_threshold(r)

  segmented_image = r > thresh

  return segmented_image

def whitehat(image,ksize=5):
  SE = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
  white_tophat_image = white_tophat(image, SE)
  return white_tophat_image

def tradition_segmentation(image,border_width=30, m=3,ksize=5):
  whitehat_img = whitehat(image,ksize)
  thresh = traditionel_threshold(whitehat_img,border_width, m)
  #thresh = triangle_threshold(image)
  segmented_image = whitehat_img > thresh
  return segmented_image








if __name__ == '__main__':
  image_path = 'point_both_38438.png'  # 替换为你的图片路径
  image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
  segmented_image = nst_star_segmentation(image,bs_size=3,bmi_size=9,bmo_size=11,be_size=11)

  a=create_matrix_with_border(7,2)
  print(a)

  # 显示原始图像和分割后的图像
  plt.figure(figsize=(10, 5))  # 设置显示图像的大小

  # 显示原始图像
  plt.subplot(1, 2, 1)
  plt.imshow(image, cmap='gray')  # 使用灰度颜色映射
  plt.title('Original Image')
  plt.axis('off')  # 不显示坐标轴

  # 显示分割后的图像
  plt.subplot(1, 2, 2)
  plt.imshow(segmented_image, cmap='gray')  # 使用灰度颜色映射
  plt.title('Segmented Image')
  plt.axis('off')  # 不显示坐标轴

  plt.show()  # 显示图像

