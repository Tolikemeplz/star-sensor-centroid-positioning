import numpy as np
import torch
from torch import nn
from torchvision.ops import nms

def pool_nms(heat, kernel = 3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

# def get_label(pred_hms,pred_offsets, gt_enlarge,confidence,cuda):
#     #   进行热力图的非极大抑制，利用3x3的卷积对热力图进行最大值筛选
#     #   找出一定区域内，得分最大的特征点。
#     pred_hms = pool_nms(pred_hms)
#
#     b, c, output_h, output_w = pred_hms.shape
#     # 用来装batch中每个样本的检测的labels
#     detects = []
#     #   只传入一张图片，循环只进行一次
#     for batch in range(b):
#         # heat_map        128*128, 1    热力图
#         # pred_offset     128*128, 2              特征点的xy轴偏移情况
#         heat_map = pred_hms[batch].permute(1, 2, 0).view([-1, c])
#         pred_offset = pred_offsets[batch].permute(1, 2, 0).view([-1, 2])
#
#         xv, yv = torch.meshgrid(torch.arange(0, output_h), torch.arange(0, output_w))
#         # xv              128*128,    特征点的x轴坐标的flatten
#         # yv              128*128,    特征点的y轴坐标的flatten
#         xv, yv = xv.flatten().float(), yv.flatten().float()
#         if cuda:
#             xv      = xv.cuda()
#             yv      = yv.cuda()
#
#         conf = heat_map.squeeze()
#         mask = conf > confidence
#
#         # 此时pred_offset_mask的形状为(num_center,2)
#         pred_offset_mask = pred_offset[mask]
#         if len(pred_offset_mask) == 0:
#             detects.append([])
#             continue
#
#         # xv_mask和yv_mask的形状都是(num_center,1),为调整后的预测星点的中心坐标
#         # 因为在dataloader文件中已经将batch_reg中的position变成放enlarge后的坐标了,所以这里直接除以gt_enlarge可得到原始坐标
#         xv_mask = torch.unsqueeze(xv[mask] + pred_offset_mask[..., 0], -1) /gt_enlarge
#         yv_mask = torch.unsqueeze(yv[mask] + pred_offset_mask[..., 1], -1) /gt_enlarge
#
#         # pred_label形状为(num_center,2)
#         # pred_label= torch.cat([xv_mask,yv_mask],dim=1)
#
#         # detect形状为(num_center,3)
#         detect = torch.cat([xv_mask,yv_mask,heat_map[mask]],dim=-1)
#         detect = detect.cpu().numpy()
#         detects.append(detect)
#
#     return detects

def get_label(pred_hms,pred_offsets, gt_enlarge,confidence,cuda):
    #   进行热力图的非极大抑制，利用3x3的卷积对热力图进行最大值筛选
    #   找出一定区域内，得分最大的特征点。
    pred_hms = pool_nms(pred_hms)

    b, c, output_h, output_w = pred_hms.shape
    # 用来装batch中每个样本的检测的labels
    detects = []
    #   只传入一张图片，循环只进行一次
    for batch in range(b):
        # heat_map        128*128, 1    热力图
        # pred_offset     128*128, 2              特征点的xy轴偏移情况
        heat_map = pred_hms[batch].permute(1, 2, 0).view([-1, c])
        pred_offset = pred_offsets[batch].permute(1, 2, 0).view([-1, 2])

        xv, yv = torch.meshgrid(torch.arange(0, output_h), torch.arange(0, output_w))
        # xv              128*128,    特征点的x轴坐标的flatten
        # yv              128*128,    特征点的y轴坐标的flatten
        xv, yv = xv.flatten().float(), yv.flatten().float()
        if cuda:
            xv      = xv.cuda()
            yv      = yv.cuda()

        conf = heat_map.squeeze()
        mask = conf > confidence

        # 此时pred_offset_mask的形状为(num_center,2)
        pred_offset_mask = pred_offset[mask]
        if len(pred_offset_mask) == 0:
            detects.append([])
            continue

        # xv_mask和yv_mask的形状都是(num_center,1),为调整后的预测星点的中心坐标
        # 因为在dataloader文件中已经将batch_reg中的position变成放enlarge后的坐标了,所以这里直接除以gt_enlarge可得到原始坐标
        xv_mask = torch.unsqueeze(xv[mask] + pred_offset_mask[..., 0], -1) /gt_enlarge
        yv_mask = torch.unsqueeze(yv[mask] + pred_offset_mask[..., 1], -1) /gt_enlarge

        # pred_label形状为(num_center,2)
        # pred_label= torch.cat([xv_mask,yv_mask],dim=1)

        # detect形状为(num_center,3)
        detect = torch.cat([xv_mask,yv_mask,heat_map[mask]],dim=-1)
        detect = detect.cpu().numpy()
        # detects.append(detect)

    return detect









def L1_bias(detect,labels):
    # labels为一个列表,每个元素是形状为(bnum_star,2)的tensor
    # detect为一个列表,每个元素是形状为(num_center,3)的tensor
    b,num_star,_ = labels.shape
    for batch in range(b):
        num_center,_ = detect[b].shape

