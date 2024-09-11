import numpy as np
from tkinter import simpledialog, Tk, Label, Button
from PIL import Image, ImageTk
import csv
import os
import matplotlib.pyplot as plt
from scipy.ndimage import fourier_shift
from scipy import ndimage
from scipy.signal import convolve2d
from scipy.fftpack import fft2, ifft2
from skimage.transform import radon
from scipy.ndimage import convolve, rotate
import cv2
from test.replicate.used import find_peak_width_corrected,z_function,double_threshold_mask,find_blur_angle
from test.replicate.wavelet import denoise_image


def improved_radon_transform(image, theta):
    """Improved Radon Transform with Z-function and double threshold mask."""
    image_z = z_function(image)
    image_mask = double_threshold_mask(image)
    image_combined = image * image_mask * image_z
    # 显示 image_z
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image_z, cmap='gray')
    plt.title('Image after Z-function')
    plt.axis('off')

    # 显示 image_combined
    plt.subplot(1, 3, 2)
    plt.imshow(image_mask, cmap='gray')
    plt.title('M mask')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(image_combined, cmap='gray')
    plt.title('Image after combining with mask and z')
    plt.axis('off')

    plt.show()
    projections = radon(image_combined, theta, circle=True)
    # print('projections 的形状为: ', projections.shape)
    return projections

# def radon_transform(image, theta):
#     projections = radon(image,theta,circle=True)
#     return  projections



def radon_transform(image, theta):
    # 获取图像尺寸
    img_center = tuple(np.array(image.shape) // 2)  # 使用整除确保结果是整数

    # 计算填充图像的大小
    max_radius = max(img_center)
    new_size = (int(2 * max_radius), int(2 * max_radius))

    # 创建一个全零的数组，用于填充
    padded_image = np.zeros(new_size, dtype=image.dtype)

    # 计算原始图像在新图像中的位置
    offset_y = int(new_size[0] / 2) - img_center[0]
    offset_x = int(new_size[1] / 2) - img_center[1]
    offset = (offset_y, offset_x)

    # 将原始图像复制到填充图像的中心
    padded_image[offset[0]:offset[0] + image.shape[0], offset[1]:offset[1] + image.shape[1]] = image

    # 执行Radon变换
    projections = radon(padded_image, theta, circle=True)

    return projections


def radon_getkernel(image,file_id,csv_file_path):
    # 执行二维傅里叶变换
    f_transformed = np.fft.fft2(image)
    f_shifted = np.fft.fftshift(f_transformed)  # 将零频率分量移到中心
    theta = np.linspace(0, 180, 180, endpoint=False)
    # projections = improved_radon_transform(np.abs(f_shifted), theta)
    projections_origin = radon_transform(np.abs(f_shifted), theta)
    blur_angle_origin, max_index_origin = find_blur_angle(projections_origin, theta)
    projection_origin = projections_origin[:, max_index_origin]

    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    # Plot the second curve
    ax.plot(range(128), projection_origin)
    ax.set_xlabel('X-axis (0 to 127)')
    ax.set_ylabel('Values')
    ax.set_title('Curve Plot of original projections')
    # Display the plots
    #plt.tight_layout()
    #plt.ion()  # 开启交互模式
    plt.show(block=True)
    # 获取用户输入
    plt.pause(0.1)  # 确保图像渲染
    D_left = None
    D_right = None

    # 检查CSV文件是否存在
    if not os.path.exists(csv_file_path):
        # 文件不存在，提示用户输入
        D_left = float(input("请输入D_left的数值: "))
        D_right = float(input("请输入D_right的数值: "))
        plt.close()
        # 创建CSV文件并写入数据
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['file_id', 'D_left', 'D_right'])
            writer.writerow([file_id, D_left, D_right])
    else:
        # 文件存在，读取数据
        file_exists = False
        with open(csv_file_path, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row['file_id'] == file_id:
                    D_left = float(row['D_left'])
                    D_right = float(row['D_right'])
                    file_exists = True
                    break
            plt.close()

        if not file_exists:
            # 文件存在但无对应的file_id，提示用户输入并添加新行
            D_left = float(input("请输入D_left的数值: "))
            D_right = float(input("请输入D_right的数值: "))
            with open(csv_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([file_id, D_left, D_right])

    #input("Press Enter to continue...")
    # D_left = float(input("请输入D_left的数值: "))
    # D_right = float(input("请输入D_right的数值: "))
    #plt.close()
    D = abs(D_right-D_left)
    return blur_angle_origin,D

def radon_getkernel_simp(image,file_id,csv_file_path):
    # 执行二维傅里叶变换
    f_transformed = np.fft.fft2(image)
    f_shifted = np.fft.fftshift(f_transformed)  # 将零频率分量移到中心
    theta = np.linspace(0, 180, 180, endpoint=False)
    # projections = improved_radon_transform(np.abs(f_shifted), theta)
    projections_origin = radon_transform(np.abs(f_shifted), theta)
    blur_angle_origin, max_index_origin = find_blur_angle(projections_origin, theta)
    projection_origin = projections_origin[:, max_index_origin]

    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    # Plot the second curve
    ax.plot(range(128), projection_origin)
    ax.set_xlabel('X-axis (0 to 127)')
    ax.set_ylabel('Values')
    ax.set_title('Curve Plot of original projections')
    # Display the plots
    #plt.tight_layout()
    #plt.ion()  # 开启交互模式
    plt.show(block=True)
    # 获取用户输入
    plt.pause(0.1)  # 确保图像渲染
    D_left = None
    D_right = None

    # 检查CSV文件是否存在
    if not os.path.exists(csv_file_path):
        # 文件不存在，提示用户输入
        D_left = float(input("请输入D_left的数值: "))
        D_right = float(input("请输入D_right的数值: "))
        plt.close()
        # 创建CSV文件并写入数据
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['file_id', 'D_left', 'D_right'])
            writer.writerow([file_id, D_left, D_right])
    else:
        # 文件存在，读取数据
        file_exists = False
        with open(csv_file_path, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row['file_id'] == file_id:
                    D_left = float(row['D_left'])
                    D_right = float(row['D_right'])
                    file_exists = True
                    break
            plt.close()

        if not file_exists:
            # 文件存在但无对应的file_id，提示用户输入并添加新行
            D_left = float(input("请输入D_left的数值: "))
            D_right = float(input("请输入D_right的数值: "))
            with open(csv_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([file_id, D_left, D_right])

    #input("Press Enter to continue...")
    # D_left = float(input("请输入D_left的数值: "))
    # D_right = float(input("请输入D_right的数值: "))
    #plt.close()
    D = abs(D_right-D_left)
    return blur_angle_origin,D








