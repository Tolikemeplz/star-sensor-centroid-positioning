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
                            calculate_weighted_centroid,generate_uniform_distribution,plot_lines_with_breaks,replace_values,calculate_line_angle,
                            average_of_third_elements,plot_tuples)
from replicate.RL import richardson_lucy_deconvolution
from replicate.wiener import wiener_filter
from replicate.NSTS import nst_star_segmentation,tradition_segmentation
from predict import predict
from replicate.NSTS import whitehat



if __name__ == "__main__":
    pic_type = 'angular'
    factor = 'magnitude'
    group=7
    start=3
    end=6
    amount = 200
    factor_list = generate_uniform_distribution(start, end, group)
    #rotate = 0
    test_name = f'{pic_type}_{factor}_{group}_{start}_{end}_{amount}'
    replicate_path = os.path.join(r'D:\programing\data\replicate', test_name)
    #factor_list = generate_uniform_distribution(start, end, amount)

    # image_path=''
    net='convnext'
    #weight = 'ep001-loss0.072-val_loss0.068.pth'
    #weight = 'ep015-loss0.067-val_loss0.088.pth'
    #weight = 'ep031-loss0.044-val_loss0.130.pth'
    #weight ='best_bias_epoch_weights.pth'
    weight='best_loss_epoch_weights.pth'
    phi=8
    confidence = 0.4
    max_centers = 5
    bias_out_path = r'D:\programing\research_code\centernet-hybrid-withoutmv-predict\test\predict_result'
    #weight_root_path = r'D:\programing\research_code\centernet-hybrid-withoutmv\logs'
    weight_root_path = r'D:\programing\data\center-hybrid_dataset_200\train_result\before_withoutmv-dropout-1'
    weight_path = os.path.join(weight_root_path, weight)

    rl_basic_group=[]
    rl_accelerated_group=[]
    wiener_group=[]
    mine_group=[]

    rl_basic_Nonenum=[]
    rl_accelerated_Nonenum=[]
    wiener_Nonenum=[]
    mine_Nonenum=[]

    for i,factor_num in enumerate(factor_list):
        group_path=os.path.join(replicate_path,f'{i}')
        image_root_path = os.path.join(group_path, 'pic')
        csv_path = os.path.join(group_path, 'csv')
        xml_root_path = os.path.join(group_path, 'annotations')

        pic_files_list = glob.glob(image_root_path + '/*.png')
        def extract_number(filename):
            # 假设文件名格式是数字_其他字符_其他字符_i.png
            # 移除文件扩展名
            base_filename = filename.split('.')[0]
            # 提取最后一个部分，假设它是序号
            return int(base_filename.split('_')[-1])
        # 使用sorted函数和lambda表达式进行排序
        pic_files_list = sorted(pic_files_list, key=extract_number)

        true_position = []
        # xi,yi,bias
        rl_basic_position = []
        rl_accelerated_position = []
        wiener_position = []
        mine_position = []

        for pic_file in pic_files_list:
            file_id = pic_file.split(".png", 1)[0]
            file_id = os.path.basename(os.path.normpath(file_id))

            print(f'factor_num为{factor_num}, file_id为：{file_id}')
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
                start_x, start_y = (float(point.find('start_x').text), float(point.find('start_y').text))
                end_x, end_y = (float(point.find('end_x').text), float(point.find('end_y').text))

            # 图片去噪
            # denoise_img = denoise_image(image,wavelet='sym17',level=3)
            denoise_img_whitehat = whitehat(image)
            denoise_img = denoise_image(image, wavelet='db8', level=3)

            # 得到真实的模糊长度和模糊角度
            l_real = sqrt((start_x - end_x) ** 2 + (start_y - end_y) ** 2)
            angle_caculate = calculate_line_angle(end_y, end_x, start_y, start_x)
            print(f'{file_id}的真实模糊角度为: {angle_caculate},模糊长度为: {l_real}')
            if l_real >= 1:
                kernel_real, _ = motion_blur_kernel(l_real, angle_caculate)
            kernel_fullsize = motion_blur_kernel_fullsize(img_shape, l_real, angle_caculate)

            # 动态复原
            rl_basic_restored = richardson_lucy_deconvolution(image, kernel_real, iterations=15,
                                                              method='basic') if l_real >= 1 else np.copy(image)
            rl_accelerated_restored = richardson_lucy_deconvolution(denoise_img_whitehat, kernel_real, iterations=15,
                                                                    method='accelerated') if l_real >= 1 else np.copy(image)
            # window = optimal_window(image, kernel_origin)
            wiener_restored = wiener_filter(denoise_img, kernel_fullsize, window=None,gamma=0.01) if l_real >= 1 else np.copy(denoise_img)

            rl_basic_segmented = tradition_segmentation(rl_basic_restored, border_width=30, m=3, ksize=5)
            rl_accelerated_segmented = tradition_segmentation(rl_basic_restored, border_width=30, m=3, ksize=5)
            wiener_segmented = tradition_segmentation(rl_basic_restored, border_width=30, m=3, ksize=5)

            # 连通域筛选
            rl_accelerated_segmented_final = process_image(rl_accelerated_segmented, rl_accelerated_restored)
            rl_basic_segmented_final = process_image(rl_basic_segmented, rl_basic_restored)
            wiener_segmented_final = process_image(wiener_segmented, wiener_restored)

            # 得到质心
            rl_accelerated_xi, rl_accelerated_yi = calculate_weighted_centroid(rl_accelerated_restored,rl_accelerated_segmented_final)
            rl_accelerated_bias = (abs(xi_true - rl_accelerated_xi) + abs(yi_true - rl_accelerated_yi)) if rl_accelerated_xi is not None else None
            if rl_accelerated_bias is not None:
                if rl_accelerated_bias > 10 :
                    rl_accelerated_bias = None
            rl_basic_xi, rl_basic_yi = calculate_weighted_centroid(rl_basic_restored, rl_basic_segmented_final)
            rl_basic_bias = (abs(xi_true - rl_basic_xi) + abs(yi_true - rl_basic_yi)) if rl_basic_xi is not None else None
            if rl_basic_bias is not None:
                if rl_basic_bias>10:
                    rl_basic_bias = None
            wiener_xi, wiener_yi = calculate_weighted_centroid(wiener_restored, wiener_segmented_final)
            wiener_bias = (abs(xi_true - wiener_xi) + abs(yi_true - wiener_yi)) if wiener_xi is not None else None
            if wiener_bias is not None:
                if wiener_bias>10:
                    wiener_bias = None
            xi, yi = predict(pic_file, weight_path, bias_out_path, confidence, max_centers, net=net, phi=phi)
            mine_bias = (abs(xi_true - xi) + abs(yi_true - yi)) if xi_true is not None else None
            if mine_bias is not None:
                if mine_bias>10:
                    mine_bias=None

            true_position.append((xi_true, yi_true))
            rl_accelerated_position.append((rl_accelerated_xi, rl_accelerated_yi, rl_accelerated_bias))
            rl_basic_position.append((rl_basic_xi, rl_basic_yi, rl_basic_bias))
            wiener_position.append((wiener_xi, wiener_yi, wiener_bias))
            mine_position.append((xi, yi, mine_bias))

            #print(f'true position: x: {xi_true} y: {yi_true}')
            print(f'rl_accelerated: x: {rl_accelerated_xi} y: {rl_accelerated_yi} bias: {rl_accelerated_bias}')
            print(f'rl_basic: x: {rl_basic_xi} y: {rl_basic_yi} bias: {rl_basic_bias}')
            print(f'wiener: x: {wiener_xi} y: {wiener_yi} bias: {wiener_bias}')
            print(f'mine: x: {xi} y: {yi} bias: {mine_bias}', '\n')

        csv_filename = f'{pic_type}_{factor}_{group}_{start}_{end}_{amount}.csv'
        csv_file_path = os.path.join(csv_path, csv_filename)

        with open(csv_file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            # 写入表头
            writer.writerow(["factor","file_id", "xi_true", "yi_true",
                             "rl_accelerated_xi", "rl_accelerated_yi", "rl_accelerated_bias",
                             "rl_basic_xi", "rl_basic_yi", "rl_basic_bias",
                             "wiener_xi", "wiener_yi", "wiener_bias",
                             "xi", "yi", "mine_bias"])
            # 写入数据
            for j, item in enumerate(pic_files_list):
                writer.writerow([factor_num,j,
                                 true_position[j][0], true_position[j][1],
                                 rl_accelerated_position[j][0], rl_accelerated_position[j][1],
                                 rl_accelerated_position[j][2],
                                 rl_basic_position[j][0], rl_basic_position[j][1], rl_basic_position[j][2],
                                 wiener_position[j][0], wiener_position[j][1], wiener_position[j][2],
                                 mine_position[j][0], mine_position[j][1], mine_position[j][2]
                                     ])

        rl_basic_bias,rl_basic_None=average_of_third_elements(rl_basic_position,amount)
        rl_accelerated_bias,rl_accelerated_None=average_of_third_elements(rl_accelerated_position,amount)
        wiener_bias,wiener_None=average_of_third_elements(wiener_position,amount)
        mine_bias,mine_None=average_of_third_elements(mine_position,amount)

        rl_basic_group.append((factor_num,rl_basic_bias))
        rl_accelerated_group.append((factor_num,rl_accelerated_bias))
        wiener_group.append((factor_num,wiener_bias))
        mine_group.append((factor_num,mine_bias))

        rl_basic_Nonenum.append((factor_num,rl_basic_None))
        rl_accelerated_Nonenum.append((factor_num,rl_accelerated_None))
        wiener_Nonenum.append((factor_num,wiener_None))
        mine_Nonenum.append((factor_num,mine_None))

    plot_tuples(rl_basic_group,rl_accelerated_group,wiener_group,mine_group)

    with open(os.path.join(replicate_path, 'rl_basic_bias.csv'), 'w', newline='') as rl_basic_file:
        writer = csv.writer(rl_basic_file)
        writer.writerow(['factor_num','bias'])  # 写入表头
        for bias in rl_basic_group:
            writer.writerow(bias)
    with open(os.path.join(replicate_path, 'rl_accelerated_bias.csv'), 'w', newline='') as rl_accelerated_file:
        writer = csv.writer(rl_accelerated_file)
        writer.writerow(['factor_num','bias'])  # 写入表头
        for bias in rl_accelerated_group:
            writer.writerow(bias)
    with open(os.path.join(replicate_path, 'wiener_bias.csv'), 'w', newline='') as wiener_file:
        writer = csv.writer(wiener_file)
        writer.writerow(['factor_num','bias'])  # 写入表头
        for bias in wiener_group:
            writer.writerow(bias)
    with open(os.path.join(replicate_path, 'mine_bias.csv'), 'w', newline='') as mine_file:
        writer = csv.writer(mine_file)
        writer.writerow(['factor_num','bias'])  # 写入表头
        for bias in mine_group:
            writer.writerow(bias)

    with open(os.path.join(replicate_path, 'rl_basic_None.csv'), 'w', newline='') as rl_basic_file:
        writer = csv.writer(rl_basic_file)
        writer.writerow(['factor_num','Nonenum'])  # 写入表头
        for Nonenum in rl_basic_Nonenum:
            writer.writerow(Nonenum)
    with open(os.path.join(replicate_path, 'rl_accelerated_None.csv'), 'w', newline='') as rl_accelerated_file:
        writer = csv.writer(rl_accelerated_file)
        writer.writerow(['factor_num','Nonenum'])  # 写入表头
        for Nonenum in rl_accelerated_Nonenum:
            writer.writerow(Nonenum)
    with open(os.path.join(replicate_path, 'wiener_None.csv'), 'w', newline='') as wiener_file:
        writer = csv.writer(wiener_file)
        writer.writerow(['factor_num','Nonenum'])  # 写入表头
        for Nonenum in wiener_Nonenum:
            writer.writerow(Nonenum)
    with open(os.path.join(replicate_path, 'mine_None.csv'), 'w', newline='') as mine_file:
        writer = csv.writer(mine_file)
        writer.writerow(['factor_num','Nonenum'])  # 写入表头
        for Nonenum in mine_Nonenum:
            writer.writerow(Nonenum)









