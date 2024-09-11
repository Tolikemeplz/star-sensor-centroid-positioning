import glob
from PIL import Image
import cv2
import os
import csv
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from replicate.radon import radon_getkernel
from replicate.wavelet import denoise_image
from replicate.used import (motion_blur_kernel,motion_blur_kernel_fullsize,optimal_window,process_image,
                            calculate_weighted_centroid,generate_uniform_distribution,plot_lines_with_breaks,replace_values,calculate_line_angle)
from replicate.RL import richardson_lucy_deconvolution
from replicate.wiener import wiener_filter
from replicate.NSTS import nst_star_segmentation,tradition_segmentation
from predict import predict
from replicate.NSTS import whitehat



if __name__ == "__main__":
    type = 'angular'
    factor = 'w'
    amount=20
    start=6
    end=7
    #rotate = 0
    test_name = f'{type}_{factor}_{amount}_{start}_{end}'
    replicate_path = os.path.join(r'D:\programing\data\replicate', test_name)
    factor_list = generate_uniform_distribution(start, end, amount)

    # image_path=''
    net='convnext'
    #weight = 'best_loss_epoch_weights.pth'
    weight = 'last_epoch_weights.pth'
    #weight ='best_bias_epoch_weights.pth'
    phi=8
    confidence = 0.4
    max_centers = 5

    image_root_path = os.path.join(replicate_path, 'pic')
    xml_root_path = os.path.join(replicate_path, 'annotations')
    weight_root_path = r'D:\programing\research_code\centernet-hybrid-withoutmv\logs'
    #weight_root_path = r'D:\programing\data\center-hybrid_dataset_200\train_result\before_withoutmv-dropout-1'
    #weight_root_path = r'D:\programing\research_code\centernet-hybrid-withoutmv\logs'
    bias_out_path = r'D:\programing\research_code\centernet-hybrid-withoutmv-predict\test\predict_result'
    weight_path = os.path.join(weight_root_path, weight)
    csv_path = os.path.join(replicate_path, 'csv')
    csv_D = csv_file_path = os.path.join(csv_path,f'{type}_{factor}_{amount}_{start}_{end}_D.csv')

    pic_files_list = glob.glob(image_root_path + '/*.png')
    # 定义一个函数，用于从文件名中提取序号i
    def extract_number(filename):
        # 假设文件名格式是数字_其他字符_其他字符_i.png
        # 移除文件扩展名
        base_filename = filename.split('.')[0]
        # 提取最后一个部分，假设它是序号
        return int(base_filename.split('_')[-1])
    # 使用sorted函数和lambda表达式进行排序
    pic_files_list = sorted(pic_files_list, key=extract_number)
    #pic_files_list.sort()

    true_position=[]
    # xi,yi,bias
    rl_basic_position=[]
    rl_accelerated_position=[]
    wiener_position=[]
    mine_position=[]


    for pic_file in pic_files_list:
        #读取图片
        #print('图片路径为：',pic_file)
        file_id = pic_file.split(".png", 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))
        print('file_id为：',file_id)
        image = cv2.imread(pic_file, cv2.IMREAD_UNCHANGED)
        img_shape = image.shape
        if image is None:
            print(f"无法读取图像：{pic_file}")

        # 从xml文件中得到真实的质心坐标
        xml_file_path = os.path.join(xml_root_path, f'{file_id}.xml')
        with open(xml_file_path) as xml_file:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            objects = root.findall('object')
            point = objects[0].find('point')
            xi_true, yi_true = (float(point.find('x').text), float(point.find('y').text))
            start_x,start_y = (float(point.find('start_x').text), float(point.find('start_y').text))
            end_x, end_y = (float(point.find('end_x').text), float(point.find('end_y').text))


        #图片去噪
        #denoise_img = denoise_image(image,wavelet='sym17',level=3)
        #denoise_img = whitehat(image)
        denoise_img = denoise_image(image, wavelet='db8', level=3)

        # 得到真实的模糊长度和模糊角度
        l_real = sqrt((start_x - end_x) ** 2 + (start_y - end_y) ** 2)
        #angle_real = 45 + rotate
        angle_caculate = calculate_line_angle(end_y, end_x, start_y, start_x)
        #print('angle_caculate :', angle_caculate)
        print(f'{file_id}的真实模糊角度为: {angle_caculate},模糊长度为: {l_real}')
        if l_real >= 1:
            kernel_real, _ = motion_blur_kernel(l_real, angle_caculate)

        # # 得到radon模糊核
        # angle, D= radon_getkernel(denoise_img,file_id,csv_D)
        # l = 2*img_shape[0]/D
        # file_id = os.path.basename(os.path.normpath(pic_file))
        # print(f'{file_id}的模糊角度为: {angle},模糊长度为: {l}')
        # kernel,kernel_origin = motion_blur_kernel(l,angle)
        kernel_fullsize = motion_blur_kernel_fullsize(img_shape,l_real,angle_caculate)

        # 动态复原
        rl_basic_restored = richardson_lucy_deconvolution(image, kernel_real, iterations=30, method='basic') if l_real>=1 else np.copy(image)
        rl_accelerated_restored = richardson_lucy_deconvolution(image, kernel_real, iterations=20, method='accelerated') if l_real>=1 else np.copy(image)
        #window = optimal_window(image, kernel_origin)
        wiener_restored = wiener_filter(denoise_img, kernel_fullsize, window=None, gamma=0.01) if l_real>=1 else np.copy(denoise_img)

        # 阈值分割
        # rl_basic_segmented = nst_star_segmentation(rl_basic_restored, bs_size=3, bmi_size=9, bmo_size=11, be_size=11)
        # rl_accelerated_segmented = nst_star_segmentation(rl_accelerated_restored, bs_size=3, bmi_size=9, bmo_size=11, be_size=11)
        # wiener_segmented = nst_star_segmentation(wiener_restored, bs_size=3, bmi_size=9, bmo_size=11, be_size=11)
        rl_basic_segmented = tradition_segmentation(rl_basic_restored, border_width=30, m=3,ksize=5)
        rl_accelerated_segmented = tradition_segmentation(rl_basic_restored, border_width=30, m=3,ksize=5)
        wiener_segmented = tradition_segmentation(rl_basic_restored, border_width=30, m=3,ksize=5)

        # 连通域筛选
        rl_accelerated_segmented_final = process_image(rl_accelerated_segmented,rl_accelerated_restored)
        rl_basic_segmented_final = process_image(rl_basic_segmented,rl_basic_restored)
        wiener_segmented_final = process_image(wiener_segmented, wiener_restored)

        # 得到质心
        rl_accelerated_xi,rl_accelerated_yi = calculate_weighted_centroid(rl_accelerated_restored,rl_accelerated_segmented_final)
        rl_accelerated_bias = (abs(xi_true-rl_accelerated_xi)+abs(yi_true-rl_accelerated_yi)) if rl_accelerated_xi is not None else None
        rl_basic_xi, rl_basic_yi = calculate_weighted_centroid(rl_basic_restored,rl_basic_segmented_final)
        rl_basic_bias = (abs(xi_true-rl_basic_xi)+abs(yi_true-rl_basic_yi)) if rl_basic_xi is not None else None
        wiener_xi, wiener_yi = calculate_weighted_centroid(wiener_restored, wiener_segmented_final)
        wiener_bias = (abs(xi_true - wiener_xi) + abs(yi_true - wiener_yi)) if wiener_xi is not None else None
        xi,yi = predict(pic_file,weight_path,bias_out_path,confidence,max_centers,net=net,phi=phi)
        mine_bias = (abs(xi_true - xi) + abs(yi_true - yi)) if xi_true is not None else None
        true_position.append((xi_true,yi_true))
        rl_accelerated_position.append((rl_accelerated_xi,rl_accelerated_yi,rl_accelerated_bias))
        rl_basic_position.append((rl_basic_xi,rl_basic_yi,rl_basic_bias))
        wiener_position.append((wiener_xi,wiener_yi,wiener_bias))
        mine_position.append((xi,yi,mine_bias))
        print(f'true position: x: {xi_true} y: {yi_true}')
        print(f'rl_accelerated: x: {rl_accelerated_xi} y: {rl_accelerated_yi} bias: {rl_accelerated_bias}')
        print(f'rl_basic: x: {rl_basic_xi} y: {rl_basic_yi} bias: {rl_basic_bias}')
        print(f'wiener: x: {wiener_xi} y: {wiener_yi} bias: {wiener_bias}')
        print(f'mine: x: {xi} y: {yi} bias: {mine_bias}','\n')

        # # 创建一个subplot，2行1列，当前选中的是第1个子图
        # plt.subplot(3, 3, 1)
        # plt.imshow(image, cmap='gray')  # 显示rl_accelerated_restored图像
        # plt.title('image')
        # plt.axis('off')  # 不显示坐标轴
        # # 创建一个subplot，2行1列，当前选中的是第1个子图
        # plt.subplot(3, 3, 2)
        # plt.imshow(denoise_img, cmap='gray')  # 显示rl_accelerated_restored图像
        # plt.title('denoise_img')
        # plt.axis('off')  # 不显示坐标轴
        # plt.subplot(3, 3, 3)
        # plt.imshow(rl_accelerated_restored, cmap='gray')  # 显示rl_accelerated_restored图像
        # plt.title('Richardson-Lucy Accelerated restored')
        # plt.axis('off')  # 不显示坐标轴
        # plt.subplot(3, 3, 4)
        # plt.imshow(rl_accelerated_segmented, cmap='gray')  # 显示rl_accelerated_restored图像
        # plt.title('Richardson-Lucy Accelerated segmented')
        # plt.axis('off')  # 不显示坐标轴
        # # 当前选中的是第2个子图
        # plt.subplot(3, 3, 5)
        # plt.imshow(wiener_restored, cmap='gray')  # 显示wiener_restored图像
        # plt.title('Wiener Filter restored')
        # plt.axis('off')  # 不显示坐标轴
        # # 当前选中的是第2个子图
        # plt.subplot(3, 3, 6)
        # plt.imshow(wiener_segmented, cmap='gray')  # 显示wiener_restored图像
        # plt.title('Wiener Filter segmented')
        # plt.axis('off')  # 不显示坐标轴
        # # 当前选中的是第2个子图
        # plt.subplot(3, 3, 7)
        # if rl_accelerated_segmented_final is not None:
        #     plt.imshow(rl_accelerated_segmented_final, cmap='gray')  # 显示wiener_restored图像
        #     plt.title('rl acc seg final')
        #     plt.axis('off')  # 不显示坐标轴
        # # 当前选中的是第2个子图
        # plt.subplot(3, 3, 8)
        # if wiener_segmented_final is not None:
        #     plt.imshow(wiener_segmented_final, cmap='gray')  # 显示wiener_restored图像
        #     plt.title('Wiener seg final')
        #     plt.axis('off')  # 不显示坐标轴
        # # 调整子图之间的间距
        # plt.tight_layout()
        # 显示图像
        #plt.show()


    # print(rl_basic_position)
    # print(rl_accelerated_position)
    # print(wiener_position)
    # print(mine_position)


    csv_filename = f'{type}_{factor}_{amount}_{start}_{end}.csv'
    csv_file_path = os.path.join(csv_path,csv_filename)

    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        # 写入表头
        writer.writerow(["factor", "xi_true", "yi_true",
                         "rl_accelerated_xi","rl_accelerated_yi","rl_accelerated_bias",
                         "rl_basic_xi", "rl_basic_yi", "rl_basic_bias",
                         "wiener_xi", "wiener_yi", "wiener_bias",
                         "xi", "yi", "mine_bias"])
        # 写入数据
        for i,item in enumerate(factor_list):
            writer.writerow([factor_list[i],
                             true_position[i][0], true_position[i][1],
                             rl_accelerated_position[i][0], rl_accelerated_position[i][1], rl_accelerated_position[i][2],
                             rl_basic_position[i][0], rl_basic_position[i][1], rl_basic_position[i][2],
                             wiener_position[i][0], wiener_position[i][1], wiener_position[i][2],
                             mine_position[i][0], mine_position[i][1], mine_position[i][2]
                             ])

    #[rl_accelerated_position, rl_basic_position, wiener_position, mine_position]=replace_values([rl_accelerated_position, rl_basic_position, wiener_position, mine_position],5)
    plot_lines_with_breaks(factor_list, [rl_accelerated_position, rl_basic_position, wiener_position, mine_position])

    with open(os.path.join(replicate_path, 'rl_accelerated_position.csv'), 'w', newline='') as rl_accelerated_file:
        writer = csv.writer(rl_accelerated_file)
        writer.writerow(['xi', 'yi','bias'])  # 写入表头
        for position in rl_accelerated_position:
            writer.writerow(position)
    with open(os.path.join(replicate_path, 'rl_basic_position.csv'), 'w', newline='') as rl_basic_file:
        writer = csv.writer(rl_basic_file)
        writer.writerow(['xi', 'yi','bias'])  # 写入表头
        for position in rl_basic_position:
            writer.writerow(position)
    with open(os.path.join(replicate_path, 'wiener_position.csv'), 'w', newline='') as wiener_file:
        writer = csv.writer(wiener_file)
        writer.writerow(['xi', 'yi','bias'])  # 写入表头
        for position in wiener_position:
            writer.writerow(position)
    with open(os.path.join(replicate_path, 'mine_position.csv'), 'w', newline='') as mine_file:
        writer = csv.writer(mine_file)
        writer.writerow(['xi', 'yi','bias'])  # 写入表头
        for position in mine_position:
            writer.writerow(position)











