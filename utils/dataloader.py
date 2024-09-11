import math
import os
# import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
# from torchvision import transforms
from utils.utils import  preprocess_input

def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(y, radius), min(width - y, radius + 1)
    top, bottom = min(x, radius), min(height - x, radius + 1)

    # masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_heatmap = heatmap[x - top:x + bottom, y - left:y + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    x, y = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_reg(batch_reg, batch_reg_mask, reg_weight, position, position_int, radius):
    diameter = 2 * radius + 1
    distance = distance2D((diameter, diameter), position, position_int)
    weight = weight2D((diameter, diameter))

    height, width = batch_reg.shape[0:2]
    # batch_reg_mask = np.zeros((height, width), dtype=np.float32)
    # reg_weight = np.zeros((height, width), dtype=np.float32)

    left, right = min(position_int[1], radius), min(width - position_int[1], radius + 1)
    top, bottom = min(position_int[0], radius), min(height - position_int[0], radius + 1)

    batch_reg_mask[position_int[0] - top:position_int[0] + bottom, position_int[1] - left:position_int[1] + right]=1

    masked_batch_reg = batch_reg[position_int[0] - top:position_int[0] + bottom, position_int[1] - left:position_int[1] + right]
    masked_distance = distance[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_batch_reg.shape) > 0 and min(masked_distance.shape) > 0:  # TODO debug
        batch_reg[position_int[0] - top:position_int[0] + bottom, position_int[1] - left:position_int[1] + right] = masked_distance

    masked_reg_weight = reg_weight[position_int[0] - top:position_int[0] + bottom, position_int[1] - left:position_int[1] + right]
    masked_weight = weight[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_reg_weight.shape) > 0 and min(masked_weight.shape) > 0:  # TODO debug
        reg_weight[position_int[0] - top:position_int[0] + bottom, position_int[1] - left:position_int[1] + right] = masked_weight


    # return batch_reg,batch_reg_mask,reg_weight

def distance2D(shape,position,position_int):
    bias = position-position_int
    m, n = [(ss - 1.) / 2. for ss in shape]
    x, y = np.ogrid[-m:m + 1, -n:n + 1]

    hx = x+bias
    hy = y+bias

    # 使用 stack 在新的轴上拼接 hx 和 hy
    result = np.stack((hx, hy), axis=-1)
    return result

def weight2D(shape):
    m, n = [(ss - 1.) / 2. for ss in shape]
    radius = max(m,n)
    x, y = np.ogrid[-m:m + 1, -n:n + 1]

    result = 1/(radius+1)/((2*np.maximum(abs(x),abs(y))+1)**2 - np.maximum( 2*np.maximum(abs(x),abs(y))-1, 0 )**2)*((2*radius+1)**2)
    return result


class HybridCenternetDataset(Dataset):
    # gt_enlarge:groundingtruth_enlarge
    def __init__(self, annotation_lines, input_shape, gt_enlarge, hm_radius, train=True, diffuse_reg=False):
        super(HybridCenternetDataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.length = len(self.annotation_lines)
        gt_enlarge = 2 **gt_enlarge
        self.input_shape = input_shape
        self.output_shape = (int(input_shape[0] * gt_enlarge), int(input_shape[1] * gt_enlarge))
        self.hm_radius = hm_radius
        self.diffuse_reg = diffuse_reg
        self.reg_radius = int(gt_enlarge / 2)
        self.train = train

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length
        line = self.annotation_lines[index].split()
        image = Image.open(line[0]).convert('I')
        image = np.array(image, np.float32)
        # 将灰度图数组形状变为(h, w, 1)
        image_np_expanded = np.expand_dims(image, axis=2)

        # 获得预测星点
        # stars的形状应该为(num_star,3)
        stars = np.array([box.split(',') for box in line[1:]])

        batch_hm = np.zeros((self.output_shape[0], self.output_shape[1], 1), dtype=np.float32)
        batch_reg = np.zeros((self.output_shape[0], self.output_shape[1], 2), dtype=np.float32)
        batch_reg_mask = np.zeros((self.output_shape[0], self.output_shape[1]), dtype=np.float32)
        reg_weight = np.zeros((self.output_shape[0], self.output_shape[1]), dtype=np.float32)


        if len(stars) != 0:
            centers = np.array(stars[:, :2], dtype=np.float32)
            centers[:, [0]] = np.clip(centers[:, [0]] / self.input_shape[0] * self.output_shape[0], 0, self.output_shape[0] - 1)
            centers[:, [1]] = np.clip(centers[:, [1]] / self.input_shape[1] * self.output_shape[1], 0, self.output_shape[1] - 1)

        for i in range (len(stars)):
            center = centers[i].copy()
            center_int = center.astype(np.int32)
            # 绘制高斯热力图
            batch_hm[:,:,0] = draw_gaussian(batch_hm[:,:,0], center_int, self.hm_radius)
            # 计算中心偏移量,并将对应的mask置为1
            # batch_reg[position_int[0], position_int[1]] = position-position_int
            # batch_reg_mask[position_int[0], position_int[1]] = 1
            if self.diffuse_reg:
                draw_reg(batch_reg, batch_reg_mask, reg_weight, center, center_int, self.reg_radius)
            else:
                batch_reg[center_int[0], center_int[1]] = center - center_int
                batch_reg_mask[center_int[0], center_int[1]] = 1
                reg_weight[center_int[0], center_int[1]] = 1
        image_np_expanded = np.transpose(preprocess_input(image_np_expanded), (2, 0, 1))

        return image_np_expanded, batch_hm, batch_reg, batch_reg_mask, reg_weight
















class CenternetDataset(Dataset):
    # gt_enlarge:groundingtruth_enlarge
    def __init__(self, csv_file, img_dir,input_shape, gt_enlarge, hm_radius, train= True, diffuse_reg=False):
        super(CenternetDataset, self).__init__()
        self.img_labels = pd.read_csv(csv_file)  # 读取 CSV 文件
        self.img_dir = img_dir
        self.input_shape = input_shape
        self.output_shape = (int(input_shape[0]*gt_enlarge) , int(input_shape[1]*gt_enlarge))
        self.hm_radius = int(hm_radius * gt_enlarge)
        self.diffuse_reg = diffuse_reg
        self.reg_radius = int(gt_enlarge/2)


    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        idx = idx % len(self.img_labels)
        # 从 CSV 文件中获取图片的文件名和标签
        img_name = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_name).convert('I')  # 假设图片是单通道的
        #image_np = np.array(image, dtype=np.float32)
        xi_label = self.img_labels.iloc[idx, 1].astype(np.float32)  # 获取 xi 标签并转换为 float32
        yi_label = self.img_labels.iloc[idx, 2].astype(np.float32)  # 获取 yi 标签并转换为 float32
        # 将标签组合成一个 NumPy 数组
        labels = np.array([xi_label, yi_label], dtype=np.float32)

        batch_hm = np.zeros((self.output_shape[0], self.output_shape[1], 1), dtype=np.float32)
        batch_reg = np.zeros((self.output_shape[0], self.output_shape[1], 2), dtype=np.float32)
        batch_reg_mask = np.zeros((self.output_shape[0], self.output_shape[1]), dtype=np.float32)
        reg_weight = np.zeros((self.output_shape[0], self.output_shape[1]), dtype=np.float32)

        # 得到ground_truth的像素坐标
        position = np.zeros(2)
        position[0] = np.clip(labels[0] / self.input_shape[0] * self.output_shape[0], 0, self.output_shape[0] - 1)
        position[1] = np.clip(labels[1] / self.input_shape[1] * self.output_shape[1], 0, self.output_shape[1] - 1)
        position_int = position.astype(np.int32)

        # 目前每张图只有一个点,所以labels形状为(1,2),蛋日后可能要检测多个星点
        labels = np.expand_dims(labels, axis=0)

        # 绘制高斯热力图
        batch_hm[:,:,0] = draw_gaussian(batch_hm[:,:,0], position_int, self.hm_radius)

        # 计算中心偏移量,并将对应的mask置为1
        # batch_reg[position_int[0], position_int[1]] = position-position_int
        # batch_reg_mask[position_int[0], position_int[1]] = 1
        if self.diffuse_reg:
            batch_reg,batch_reg_mask,reg_weight = draw_reg(batch_reg,position,position_int,self.reg_radius)
        else:
            batch_reg[position_int[0], position_int[1]] = position - position_int
            batch_reg_mask[position_int[0], position_int[1]] = 1
            reg_weight[position_int[0], position_int[1]] = 1


        image = np.transpose(preprocess_input(image), (2, 0, 1))

        return image, batch_hm, batch_reg, batch_reg_mask, labels, reg_weight

def centernet_dataset_collate(batch):
    imgs, batch_hms,batch_regs, batch_reg_masks, reg_weights,= [], [], [], [], []

    for img, batch_hm, batch_reg, batch_reg_mask, reg_weight in batch:
        imgs.append(img)
        batch_hms.append(batch_hm)
        batch_regs.append(batch_reg)
        batch_reg_masks.append(batch_reg_mask)
        reg_weights.append(reg_weight)

    imgs = torch.from_numpy(np.array(imgs)).type(torch.FloatTensor)
    batch_hms = torch.from_numpy(np.array(batch_hms)).type(torch.FloatTensor)
    batch_regs = torch.from_numpy(np.array(batch_regs)).type(torch.FloatTensor)
    batch_reg_masks = torch.from_numpy(np.array(batch_reg_masks)).type(torch.FloatTensor)
    reg_weights = torch.from_numpy(np.array(reg_weights)).type(torch.FloatTensor)
    return imgs, batch_hms, batch_regs, batch_reg_masks, reg_weights















