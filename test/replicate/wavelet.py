import numpy as np
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt
import pywt
import copy

def normal_shrink(coeffs):
  num_levels = len(coeffs) - 1
  cD1 = coeffs[-1][2]
  sigma = np.std(cD1)
  denoised_coeffs = copy.deepcopy(coeffs)

  for level in range(num_levels):
    # 获取当前尺度的子带系数
    cH, cV, cD = coeffs[level+1]
    # 计算阈值
    L,W = cH.shape
    beta = np.sqrt(np.log(L*W/num_levels))
    sigma_y_cH = np.std(cH)
    sigma_y_cV = np.std(cV)
    sigma_y_cD = np.std(cD)
    T_cH = beta * sigma ** 2 / sigma_y_cH
    T_cV = beta * sigma ** 2 / sigma_y_cV
    T_cD = beta * sigma ** 2 / sigma_y_cD

    # 应用软阈值去噪
    cH_denoised = np.where(np.abs(cH) > T_cH, cH, 0)
    cV_denoised = np.where(np.abs(cV) > T_cV, cV, 0)
    cD_denoised = np.where(np.abs(cD) > T_cD, cD, 0)

    # 将去噪后的系数存储到结果中
    denoised_coeffs[level+1] = (cH_denoised, cV_denoised, cD_denoised)

  # 返回去噪后的小波系数
  return denoised_coeffs

def denoise_image(image,wavelet,level):
  # 进行小波分解
  coeffs = pywt.wavedec2(image, wavelet, level=level)

  # 应用 NormalShrink 去噪算法
  denoised_coeffs = normal_shrink(coeffs)

  # 进行小波重构
  denoised_image = pywt.waverec2(denoised_coeffs, wavelet)

  return denoised_image

if __name__ == '__main__':
  # 示例用法
  # 加载图像
  image_path = '1_angular_both_3.png'  # 替换为你的图片路径
  image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

  # 添加噪声
  sigma_noise = 60
  noisy_image = image + sigma_noise * np.random.randn(*image.shape)+100
  noisy_image = np.clip(noisy_image, 0, 1050)
  noisy_image = image
  # 应用 NormalShrink 去噪算法
  denoised_image = denoise_image(noisy_image,wavelet='sym17',level=3)

  # 计算噪声图像（原始图像与去噪图像之差）
  noise_image = image - denoised_image
  # 计算原始图像的功率
  P_signal = np.sum(image**2)
  # 计算噪声的功率
  P_noise = np.sum(noise_image**2)
  # 计算信噪比
  SNR = 10 * np.log10(P_signal / P_noise)
  # 打印信噪比
  print(f"The Signal-to-Noise Ratio (SNR) is: {SNR} dB")


  # 显示原始图片
  plt.subplot(1, 3, 1)  # 1行2列的第一个位置
  plt.imshow(image, cmap='gray')  # 确保是灰度图
  plt.title('Original Image')
  plt.axis('off')  # 不显示坐标轴

  # 显示噪声图片
  plt.subplot(1, 3, 2)  # 1行2列的第一个位置
  plt.imshow(noisy_image, cmap='gray')  # 确保是灰度图
  plt.title('noisy Image')
  plt.axis('off')  # 不显示坐标轴

  # 显示去噪后的图片
  plt.subplot(1, 3, 3)  # 1行2列的第二个位置
  plt.imshow(denoised_image, cmap='gray')  # 确保是灰度图
  plt.title('Denoised Image')
  plt.axis('off')  # 不显示坐标轴

  plt.show()  # 显示图像

  # for wavelet in ['db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'db9', 'db10', 'db11', 'db12', 'db13', 'db14',
  #                 'db15', 'db16', 'db17', 'db18', 'db19', 'db20']:
  #   # 应用 NormalShrink 去噪算法
  #   denoised_image = denoise_image(noisy_image, wavelet=wavelet, level=3)
  #
  #   # 计算噪声图像（原始图像与去噪图像之差）
  #   noise_image = image - denoised_image
  #   # 计算原始图像的功率
  #   P_signal = np.sum(image ** 2)
  #   # 计算噪声的功率
  #   P_noise = np.sum(noise_image ** 2)
  #   # 计算信噪比
  #   SNR = 10 * np.log10(P_signal / P_noise)
  #   # 打印信噪比
  #   print(f"The Signal-to-Noise Ratio (SNR) using {wavelet} wavelet is: {SNR} dB")
  #
  # for wavelet in ['sym2', 'sym3', 'sym4', 'sym5', 'sym6', 'sym7', 'sym8', 'sym9', 'sym10', 'sym11', 'sym12', 'sym13',
  #                 'sym14', 'sym15', 'sym16', 'sym17', 'sym18', 'sym19', 'sym20']:
  #   # 应用 NormalShrink 去噪算法
  #   denoised_image = denoise_image(noisy_image, wavelet=wavelet, level=3)
  #
  #   # 计算噪声图像（原始图像与去噪图像之差）
  #   noise_image = image - denoised_image
  #   # 计算原始图像的功率
  #   P_signal = np.sum(image ** 2)
  #   # 计算噪声的功率
  #   P_noise = np.sum(noise_image ** 2)
  #   # 计算信噪比
  #   SNR = 10 * np.log10(P_signal / P_noise)
  #   # 打印信噪比
  #   print(f"The Signal-to-Noise Ratio (SNR) using {wavelet} wavelet is: {SNR} dB")



