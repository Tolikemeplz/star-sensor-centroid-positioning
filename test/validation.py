import glob
import json
import math
import operator
import os
import shutil
import csv
import sys
import cv2
import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from model.hybrid_centernet import Convnext_Centernet
from model_function import Model_function
from PIL import Image
from tqdm import tqdm
from valid_utils import get_dic,bias_plot,add_dic,plot_and_save,plot_bar_and_save,plot_and_save_objects
from extract_dataset import filter_lines_by_keyword,apart_lines_by_factor


if __name__ == "__main__":
    # mode=0:评估整个keyword数据集和每个apart组数据集
    # mode=1:只评估整个keyword数据集
    # mode=2:只评估每个每个apart组数据集
    # mode=3:只评估每个type组数据集
    mode=2
    gt_enlarge = 0
    phi = 8
    confidence = 0.05
    MAXBIAS = 1
    score_threhold = 0.4
    pic_type = 'angular'
    keyword = 'angular_both'
    factor = 'wxy'
    apart = 10
    part = 40

    #model_path = r'D:\programing\data\center-hybrid_dataset_200\train_result\before_withoutmv-13\best_loss_epoch_weights.pth'
    model_path = r'D:\programing\data\center-hybrid_dataset_200\train_result\before_withoutmv-dropout-1\best_loss_epoch_weights.pth'
    #model_path = r'D:\programing\data\center-hybrid_dataset_200\train_result\before_withoutmv-dropout-1\last_epoch_weights.pth'
    RESULT_PATH = r'D:\programing\research_code\centernet-hybrid-withoutmv-predict\test\result'
    # 存放每个apart组的验证结果
    APART_RESULT_PATH = r'D:\programing\research_code\centernet-hybrid-withoutmv-predict\test\result\apart_result'
    # 存放每个type的验证结果
    TYPE_RESULT_PATH =r'D:\programing\research_code\centernet-hybrid-withoutmv-predict\test\result\type_result'
    CSV_PATH = r'D:\programing\data\center-hybrid_dataset_200\csv\val.csv'
    source_file_path = r'D:\programing\research_code\centernet-hybrid-withoutmv-predict\dataset_200_val.txt'

    model_function = Model_function(model_path=model_path, confidence=confidence, phi=phi)

    if keyword !='':
        val_annotation_path = filter_lines_by_keyword(source_file_path=source_file_path,
                                output_dir=r'D:\programing\research_code\centernet-hybrid-withoutmv-predict\test',
                                keyword=keyword)

    if mode == 0 or mode == 2:
        # group_paths装有是annotation_apart文件夹中所有txt文件地址的一个列表
        group_midpoints, group_paths = apart_lines_by_factor(source_file_path=val_annotation_path,
                                                             output_dir=r'D:\programing\research_code\centernet-hybrid-withoutmv-predict\test\annotation_apart',
                                                             csv_path=CSV_PATH,
                                                             factor=factor,
                                                             apart=apart)
        valid_apart_out_path = ".temp_valid_apart_out"
        if not os.path.exists(valid_apart_out_path):
            os.makedirs(valid_apart_out_path)
        group_ap = []
        group_F1 = []
        group_rec = []
        group_prec = []
        for i,group_path in enumerate(group_paths):
            # apart_result文件夹中以group_midpoints[i]命名的文件夹的地址
            group_result_path = os.path.join(APART_RESULT_PATH, f"{group_midpoints[i]}")
            if not os.path.exists(group_result_path):
                os.makedirs(group_result_path)
            # 存放group_midpoints[i]这一组的临时文件的地址
            apart_path = os.path.join(valid_apart_out_path, f"{group_midpoints[i]}")
            if not os.path.exists(apart_path):
                os.makedirs(apart_path)
            # 这一个group的所有样本的annotation_line
            with open(group_path) as f:
                group_lines = f.readlines()

            # 创建两个子目录"ground-truth"和"detection-results"，用于存储真实框和检测结果的文本文件。
            if not os.path.exists(os.path.join(apart_path, "ground-truth")):
                os.makedirs(os.path.join(apart_path, "ground-truth"))
            if not os.path.exists(os.path.join(apart_path, "detection-results")):
                os.makedirs(os.path.join(apart_path, "detection-results"))

            print(f"Get apart ({group_midpoints[i]}) ground-truth and detection-results txt")
            for annotation_line in tqdm(group_lines):
                line = annotation_line.split()
                image_id = os.path.basename(line[0]).split('.')[0]
                image = Image.open(line[0]).convert('I')  # 假设图片是单通道的
                # 获得真实坐标
                gt_centers = np.array([np.array(list(map(float, box.split(',')))) for box in line[1:]])
                # 获得真实坐标txt
                with open(os.path.join(apart_path, "ground-truth/" + image_id + ".txt"), "w") as new_f:
                    for center in gt_centers:
                        xi, yi, mv = center
                        new_f.write("%s %s %s\n" % (xi, yi, mv))
                # 获得检测坐标txt
                model_function.get_map_txt(image_id, image, apart_path)
            print(f"Get apart ({group_midpoints[i]}) predict result and ground truth result done.")
            ap,F1,rec,prec,_=get_dic(
                MAXBIAS=MAXBIAS, RESULT_PATH=group_result_path, CSV_PATH=CSV_PATH, score_threhold=score_threhold,
                    path=apart_path, is_apart=True)
            group_ap.append((group_midpoints[i],ap))
            group_F1.append((group_midpoints[i],F1))
            group_rec.append((group_midpoints[i], rec))
            group_prec.append((group_midpoints[i], prec))

        # 如果保存路径不存在，则创建它
        if not os.path.exists(APART_RESULT_PATH):
            os.makedirs(APART_RESULT_PATH)
        # 生成图表并保存
        plot_and_save(group_ap, 'AP Scores by Factor', 'ap_scores.png', APART_RESULT_PATH)
        plot_and_save(group_F1, 'F1 Scores by Factor', 'f1_scores.png', APART_RESULT_PATH)
        plot_and_save(group_rec, 'Recall Scores by Factor', 'recall_scores.png', APART_RESULT_PATH)
        plot_and_save(group_prec, 'Precision Scores by Factor', 'precision_scores.png', APART_RESULT_PATH)
        plot_and_save_objects(group_rec,group_prec,group_F1,'w (°/s)','Objective detection metrics',
                              'Objective detection metrics.png', APART_RESULT_PATH, 2)
        # 存储group_F1为CSV文件
        with open(os.path.join(APART_RESULT_PATH, 'f1_scores.csv'), 'w', newline='') as f1_file:
            writer = csv.writer(f1_file)
            writer.writerow(['Group', 'F1'])  # 写入表头
            for group in group_F1:
                writer.writerow(group)

        # 存储group_rec为CSV文件
        with open(os.path.join(APART_RESULT_PATH, 'recall_scores.csv'), 'w', newline='') as rec_file:
            writer = csv.writer(rec_file)
            writer.writerow(['Group', 'Recall'])  # 写入表头
            for group in group_rec:
                writer.writerow(group)

        # 存储group_prec为CSV文件
        with open(os.path.join(APART_RESULT_PATH, 'precision_scores.csv'), 'w', newline='') as prec_file:
            writer = csv.writer(prec_file)
            writer.writerow(['Group', 'Precision'])  # 写入表头
            for group in group_prec:
                writer.writerow(group)

        shutil.rmtree(valid_apart_out_path)


    if mode == 3:
        type_list = ['point', 'angular', 'linear', 'power', 'bezier']
        valid_out_path = ".temp_valid_type_out"
        if not os.path.exists(valid_out_path):
            os.makedirs(valid_out_path)
        type_ap = []
        type_F1 = []
        type_rec = []
        type_prec = []
        type_bias = []
        for i,curve_type in enumerate(type_list):
            val_annotation_path = filter_lines_by_keyword(source_file_path=source_file_path,
                                                          output_dir=r'D:\programing\research_code\centernet-hybrid-withoutmv-predict\test',
                                                          keyword=curve_type)
            type_result_path = os.path.join(TYPE_RESULT_PATH,curve_type)
            if not os.path.exists(type_result_path):
                os.makedirs(type_result_path)
            type_temp_path = os.path.join(valid_out_path,curve_type)
            if not os.path.exists(type_temp_path):
                os.makedirs(type_temp_path)
            # 这一个group的所有样本的annotation_line
            with open(val_annotation_path) as f:
                type_lines = f.readlines()
            # 创建两个子目录"ground-truth"和"detection-results"，用于存储真实框和检测结果的文本文件。
            if not os.path.exists(os.path.join(type_temp_path, "ground-truth")):
                os.makedirs(os.path.join(type_temp_path, "ground-truth"))
            if not os.path.exists(os.path.join(type_temp_path, "detection-results")):
                os.makedirs(os.path.join(type_temp_path, "detection-results"))

            print(f"Get type: {curve_type} ground-truth and detection-results txt")
            for annotation_line in tqdm(type_lines):
                line = annotation_line.split()
                image_id = os.path.basename(line[0]).split('.')[0]
                image = Image.open(line[0]).convert('I')  # 假设图片是单通道的
                # 获得真实坐标
                gt_centers = np.array([np.array(list(map(float, box.split(',')))) for box in line[1:]])
                # 获得真实坐标txt
                with open(os.path.join(type_temp_path, "ground-truth/" + image_id + ".txt"), "w") as new_f:
                    for center in gt_centers:
                        xi, yi, mv = center
                        new_f.write("%s %s %s\n" % (xi, yi, mv))
                # 获得检测坐标txt
                model_function.get_map_txt(image_id, image, type_temp_path)
            print(f"Get type {curve_type} predict result and ground truth result done.")
            ap, F1, rec, prec, bias = get_dic(
                MAXBIAS=MAXBIAS, RESULT_PATH=type_result_path, CSV_PATH=CSV_PATH, score_threhold=score_threhold,
                path=type_temp_path, is_apart=True)
            type_ap.append((type_list[i],ap))
            type_F1.append((type_list[i],F1))
            type_rec.append((type_list[i],rec))
            type_prec.append((type_list[i],prec))
            type_bias.append((type_list[i],bias))

        if not os.path.exists(TYPE_RESULT_PATH):
            os.makedirs(TYPE_RESULT_PATH)
        plot_bar_and_save(type_ap, 'AP Scores by type', 'ap_scores.png', TYPE_RESULT_PATH)
        plot_bar_and_save(type_F1, 'F1 Scores by type', 'F1_scores.png', TYPE_RESULT_PATH)
        plot_bar_and_save(type_rec, 'rec Scores by type', 'rec_scores.png', TYPE_RESULT_PATH)
        plot_bar_and_save(type_prec, 'prec Scores by type', 'prec_scores.png', TYPE_RESULT_PATH)
        plot_bar_and_save(type_bias, 'bias Scores by type', 'bias_scores.png', TYPE_RESULT_PATH)
        shutil.rmtree(valid_out_path)








    if mode ==0 or mode == 1:
        valid_out_path = ".temp_valid_out"
        if not os.path.exists(valid_out_path):
            os.makedirs(valid_out_path)

        with open(val_annotation_path) as f:
            val_lines   = f.readlines()

        # 创建两个子目录"ground-truth"和"detection-results"，用于存储真实框和检测结果的文本文件。
        if not os.path.exists(os.path.join(valid_out_path, "ground-truth")):
            os.makedirs(os.path.join(valid_out_path, "ground-truth"))
        if not os.path.exists(os.path.join(valid_out_path, "detection-results")):
            os.makedirs(os.path.join(valid_out_path, "detection-results"))

        print("Get ground-truth and detection-results txt")
        for annotation_line in tqdm(val_lines):
            line = annotation_line.split()
            image_id = os.path.basename(line[0]).split('.')[0]
            image = Image.open(line[0]).convert('I')  # 假设图片是单通道的
            # 获得真实坐标
            gt_centers = np.array([np.array(list(map(float, box.split(',')))) for box in line[1:]])
            # 获得真实框txt
            with open(os.path.join(valid_out_path, "ground-truth/" + image_id + ".txt"), "w") as new_f:
                for center in gt_centers:
                    xi, yi, mv = center
                    new_f.write("%s %s %s\n" % (xi, yi, mv))

            model_function.get_map_txt(image_id, image,  valid_out_path)
        print("Get predict result and ground truth result done.")

        get_dic(MAXBIAS=MAXBIAS,RESULT_PATH=RESULT_PATH,CSV_PATH=CSV_PATH,score_threhold=score_threhold,path=valid_out_path,)

        add_dic(RESULT_PATH,CSV_PATH,pic_type)

        bias_plot(factor,RESULT_PATH,part=part,smooth=False)

        shutil.rmtree(valid_out_path)



