import numpy as np
import cv2
from numpy import fft
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from test.replicate.used import motion_blur_kernel, optimal_window, motion_blur_kernel_win
from test.replicate.NSTS import nst_star_segmentation
from test.replicate.threshold import imply_threshold,imply_triangle_threshold

def wiener_filter(image, psf, window=None, gamma=0.001, eps=1e-4):
  # 确保输入图像是浮点数类型
  image = np.float32(image)

  # 计算最优窗加权的模糊图像
  if window is not None:
    # 确保窗函数与图像尺寸匹配
    assert window.shape == image.shape, "Window size must match image size."
    weighted_image = image * window
  else:
    weighted_image = image

  # 进行FFT
  f = fft2(weighted_image)

  # 计算点扩散函数的FFT
  h = fft2(psf) + eps

  # 计算维纳滤波器
  h_conj = np.conj(h)
  denominator = np.abs(h) ** 2 + gamma
  filter = h_conj / denominator

  # 应用维纳滤波器
  f_filtered = f * filter

  # 计算恢复后的图像的傅里叶逆变换
  restored_image = ifft2(f_filtered)

  # 返回恢复后的图像的实部
  return np.abs(fftshift(restored_image))


def wiener(input, PSF, eps, K=0.01):  # 维纳滤波，K=0.01
  input_fft = fft.fft2(input)
  PSF_fft = fft.fft2(PSF) + eps
  PSF_fft_1 = np.conj(PSF_fft) / (np.abs(PSF_fft) ** 2 + K)
  result = fft.ifft2(input_fft * PSF_fft_1)
  result = np.abs(fft.fftshift(result))
  return result



if __name__ == '__main__':
  image_path = 'point_None_9994.png'  # 替换为你的图片路径
  image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
  # 假设 image.shape 是 (M, N)
  img_shape = image.shape

  # 设置模糊参数
  blur_length = 40  # 模糊长度，根据需要调整
  blur_angle = 137  # 模糊角度，单位为度

  # 应用运动模糊
  blurred_image = apply_motion_blur(image, blur_length, blur_angle)
  noisy_image = add_gaussian_noise(blurred_image, mean=40, sigma=25)
  kernel = motion_blur_kernel(blur_length, blur_angle)
  kernel_win = motion_blur_kernel_win(blur_length,blur_angle)

  window = optimal_window(blurred_image,kernel_win)
  restored_image = wiener_filter(noisy_image,kernel,window,gamma=0.01)
  #restored_image =wiener(blurred_image, kernel, eps=1e-8,K=0.001)

  segmented_image = nst_star_segmentation(restored_image,bs_size=3,bmi_size=9,bmo_size=11,be_size=11)
  #segmented_image = imply_threshold(restored_image,m=3)
  #segmented_image = imply_triangle_threshold(restored_image)


  # 创建一个1x3的子图网格
  fig, ax = plt.subplots(1, 4, figsize=(15, 5))

  # 显示原始图像
  ax[0].imshow(image, cmap='gray')  # 使用灰度颜色映射
  ax[0].set_title('Original Image')
  ax[0].axis('off')  # 关闭坐标轴

  # 显示模糊图像
  ax[1].imshow(noisy_image, cmap='gray')  # 使用灰度颜色映射
  ax[1].set_title('Noisy Image')
  ax[1].axis('off')

  # 显示恢复后的图像
  ax[2].imshow(restored_image, cmap='gray')  # 使用灰度颜色映射
  ax[2].set_title('Restored Image')
  ax[2].axis('off')

  ax[3].imshow(segmented_image, cmap='gray')  # 使用灰度颜色映射
  ax[3].set_title('segmented Image')
  ax[3].axis('off')

  # 显示图形
  plt.show()




