import numpy as np
import cv2
from scipy.signal import convolve2d
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
from test.replicate.used import motion_blur_kernel
from test.replicate.NSTS import nst_star_segmentation
from test.replicate.wavelet import denoise_image

def richardson_lucy_deconvolution(image, psf, iterations=10, method='basic'):
    if method != 'accelerated':
        return basic_richardson_lucy(image, psf, iterations)
        #return RL_deconvblind(image,psf,iterations)
    else:
        return accelerated_richardson_lucy(image, psf, iterations)



def accelerated_richardson_lucy(image, psf, iterations):
    est_prev = np.copy(image)
    est = np.copy(image)
    d1 = np.zeros_like(image)
    d2 = np.zeros_like(image)

    for i in range(iterations):
        if i < 2:  # No acceleration for first two iterations
            est = basic_richardson_lucy_step_adjust(est_prev, image, psf)
        else:
            # 确保在进行减法操作前，所有的值都是有效的
            est = np.nan_to_num(est)
            est_prev = np.nan_to_num(est_prev)
            est_prev_prev = np.nan_to_num(est_prev_prev)
            alpha = compute_alpha(est_prev - est_prev_prev, est - est_prev)
            d1 = est - est_prev
            d2 = est - 2 * est_prev + est_prev_prev
            extrapolated_est = est + alpha * d1 + 0.5 * alpha ** 2 * d2
            est = basic_richardson_lucy_step_adjust(extrapolated_est, image, psf)

        est_prev_prev, est_prev = est_prev, est
    # 缩放像素值到0到254之间
    # restored_image = est.clip(0, 254).astype(np.uint8)
    return est

def basic_richardson_lucy(image, psf, iterations):
    # 确保estimate是浮点类型
    estimate = image.astype(np.float64)
    image = image.astype(np.float64)
    psf = psf.astype(np.float64)
    #print('psf为：\n',psf)

    for i in range(iterations):
        # 1. 对估计图像进行卷积
        conv_estimate = convolve(estimate, psf, mode='mirror')
        # 避免除以零
        conv_estimate[conv_estimate == 0] = np.finfo(float).eps
        # 2. 计算比率
        ratio = image / (conv_estimate)
        # 3. 对比率进行卷积
        conv_ratio = convolve(ratio, np.rot90(psf, 2), mode='mirror')
        # 避免除以零
        conv_ratio[conv_ratio == 0] = np.finfo(float).eps
        #print(f'第{i+1}次迭代的conv_ratio中的最大值为为 {np.max(conv_ratio)}')
        # 4. 更新估计图像
        estimate *= conv_ratio
    # 缩放像素值到0到254之间
    #restored_image = estimate.clip(0, 254).astype(np.uint8)
    restored_image = estimate
    return restored_image



def basic_richardson_lucy_step_adjust(estimate, image, psf):
    # 确保estimate是浮点类型
    max_val = 1e30  # 设置一个合理的最大值
    estimate = estimate.astype(np.float64)
    image = image.astype(np.float64)
    # 1. 对估计图像进行卷积
    conv_estimate = convolve(estimate, psf, mode='mirror')
    # 避免除以零
    conv_estimate[conv_estimate == 0] = np.finfo(float).eps
    # 2. 计算比率
    ratio = image / (conv_estimate)
    # 3. 对比率进行卷积
    conv_ratio = convolve(ratio, np.rot90(psf, 2), mode='mirror')
    # 避免除以零
    conv_ratio[conv_ratio == 0] = np.finfo(float).eps
    # 4. 更新估计图像
    # estimate = np.clip(estimate, 0, max_val)
    # conv_ratio = np.clip(conv_ratio, 0, max_val)
    estimate *= conv_ratio
    return estimate



def compute_alpha(g_k_minus_1, g_k_minus_2):
    # Simplified calculation of alpha, assumes g_k_minus_1 and g_k_minus_2 are not zero
    max_val = 1e30  # 设置一个合理的最大值
    g_k_minus_1 = np.clip(g_k_minus_1, -max_val, max_val)
    g_k_minus_2 = np.clip(g_k_minus_2, -max_val, max_val)
    gkT_gk_minus_1 = np.sum(g_k_minus_1 * g_k_minus_2)
    gk_minus_1T_gk_minus_1 = np.sum(np.square(g_k_minus_1))
    alpha = gkT_gk_minus_1 / (gk_minus_1T_gk_minus_1 + 1e-12)
    return alpha



# Example usage
if __name__ == '__main__':
    # Assuming 'image' and 'psf' are pre-loaded NumPy arrays
    # 'image' is the blurred image, 'psf' is the point spread function (blur kernel)
    # 读取图片
    image_path = 'point_None_9994.png'  # 替换为你的图片路径
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # blurred_star = simulate_blur_gaussian(star_image, 10, 45)

    # 设置模糊参数
    blur_length = 31.5 # 模糊长度，根据需要调整
    blur_angle = 45  # 模糊角度，单位为度

    # 应用运动模糊
    blurred_image = apply_motion_blur_new(image, blur_length, blur_angle)
    noisy_image = add_gaussian_noise(blurred_image, mean=10, sigma=30)
    # noisy_image = denoise_image(noisy_image)

    kerner = motion_blur_kernel(blur_length, blur_angle)
    kerner_new = motion_blur_kernel_new(blur_length,blur_angle)
    # kerner = kerner.astype(np.float64) / np.sum(kerner)  # 归一化PSF
    print('kernel shape: ', kerner_new.shape)

    # method: basic accelerated
    restored_image = richardson_lucy_deconvolution(noisy_image, kerner, iterations=15, method='accelerated')
    # Do something with the restored image

    segmented_image = nst_star_segmentation(restored_image, bs_size=3, bmi_size=9, bmo_size=11, be_size=11)

    # 显示恢复后的图像
    plt.figure(figsize=(10, 5))  # 设置显示窗口的大小

    # 显示原始图像
    plt.subplot(2, 2, 1)
    plt.imshow(image, cmap='gray')  # 假设图像是灰度的，如果不是，请移除 cmap='gray'
    plt.title('Original Image')
    plt.axis('off')  # 不显示坐标轴

    # 显示模糊图像
    plt.subplot(2, 2, 2)
    plt.imshow(noisy_image, cmap='gray')  # 假设图像是灰度的
    plt.title('noisy Image')
    plt.axis('off')

    # 显示恢复后的图像
    plt.subplot(2, 2, 3)
    plt.imshow(restored_image, cmap='gray')  # 假设图像是灰度的
    plt.title('Restored Image')
    plt.axis('off')

    # 显示恢复后的图像
    plt.subplot(2, 2, 4)
    plt.imshow(segmented_image, cmap='gray')  # 假设图像是灰度的
    plt.title('Segmented_image')
    plt.axis('off')

    plt.show()  # 显示图像
