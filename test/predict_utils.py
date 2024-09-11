import os

import matplotlib
import torch
from torch import nn

#matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import scipy.signal

import shutil
import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def preprocess_input(image):
    image   = np.array(image,dtype = np.float32)
    mean    = 0.08330865
    std     = 0.063446214
    return (image /1023 - mean) / std

def pool_nms(heat, kernel = 3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def get_label(pred_hms,pred_offsets, gt_enlarge,confidence,cuda):
    #   进行热力图的非极大抑制，利用3x3的卷积对热力图进行最大值筛选
    #   找出一定区域内，得分最大的特征点。
    pred_hms = pool_nms(pred_hms)
    gt_enlarge = 2**gt_enlarge

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



def get_map_txt(image_id, net, image, bias_out_path,confidence,max_centers,gt_enlarge):
    f = open(os.path.join(bias_out_path, image_id + ".txt"), "w")
    # 计算输入图片的高和宽
    image_shape = np.array(np.shape(image)[0:2])
    #print('image_shape:',np.shape(image))
    # 因为输入的image是Image对象,只有2维
    image_np_expanded = np.expand_dims(image, axis=2)

    #   图片预处理，归一化。获得的photo的shape为[1, 1, 128, 128]
    image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_np_expanded, dtype='float32')), (2, 0, 1)),
                                0)

    with torch.no_grad():
        images = torch.from_numpy(np.asarray(image_data)).type(torch.FloatTensor)
        images = images.cuda()
        #   将图像输入网络当中进行预测！
        outputs = net(images)

        # 检查outputs中是否有inf值
        inf_in_outputs = torch.isinf(outputs[0])
        #print('outputs[0]的形状为： ',outputs[0].shape)
        # 如果inf_in_outputs是tensor，那么它将包含一个布尔mask，指示哪些位置是inf
        # 我们可以统计inf值的数量
        # inf_count = inf_in_outputs.sum()
        # if inf_count > 0:
        #     print(f"outputs contains {inf_count} inf values")
        # else:
        #     print("outputs does not contain any inf values")

        #   利用预测结果进行解码
        output = get_label(outputs[0], outputs[1], gt_enlarge, confidence, cuda=True)
        # 使用np.isinf检查output数组中是否有inf值
        inf_mask = np.isinf(output)
        # 计算inf值的数量
        inf_count = np.sum(inf_mask)
        # 打印inf值的数量
        #print(f"Number of inf values in output: {inf_count}")

        # 这里用result[0]可能是因为只传入了一张图片,所以只取第一张
        if output is None:
            return

        # 置信度
        top_conf = output[:, 2]
        # 预测星点坐标
        top_centers = output[:, :2]

    # 根据置信度对预测框进行排序,并选择前self.max_boxes个最高的置信度框。
    top_100 = np.argsort(top_conf)[::-1][:max_centers]
    #  然后, 根据这个排序, 重新选择对应的边界狂,置信度和标签
    top_centers = top_centers[top_100]
    top_conf = top_conf[top_100]

    for i, c in list(enumerate(top_conf)):
        centers = top_centers[i]
        #print('坐标为： ',centers)
        score = str(top_conf[i])

        xi, yi = centers

        # 这表示变量score（假设是一个字符串）的前8个字符的切片
        f.write("%s %s %s\n" % (score[:8], str(xi), str(yi)))

    f.close()

    image_unit16 = np.array(image).astype(np.uint16)
    heatmap_unit16 = (outputs[0].cpu().numpy().squeeze()*1024).astype(np.uint16)
    #print('heatmap_shape::',heatmap_unit16.shape)
    pil_image = Image.fromarray(image_unit16, mode='I;16')
    pil_heatmap = Image.fromarray(heatmap_unit16, mode='I;16')
    pic_path = os.path.join(bias_out_path,'pic')
    if not os.path.exists(pic_path):
        os.makedirs(pic_path)
    img_path = os.path.join(bias_out_path,'pic', f'{image_id}.png')
    heatmap_path = os.path.join(bias_out_path, 'pic', f'{image_id}_heatmap.png')
    pil_image.save(img_path)
    pil_heatmap.save(heatmap_path)


    # 显示图像和outputs[0]的热力图
    plt.figure(figsize=(6, 6))  # 设置画布大小
    # 显示原始图像
    #plt.subplot(1, 2, 1)
    plt.imshow(np.array(image), cmap='gray')
    #plt.title('Original Image')
    plt.axis('off')  # 不显示坐标轴
    plt.show()  # 显示原始图像

    # 显示outputs[0]的热力图
    #plt.subplot(1, 2, 2)
    output_data = outputs[0].cpu().numpy().squeeze()  # 去掉不必要的维度
    plt.imshow(output_data, cmap='viridis')  # 使用热力图颜色映射
    #plt.title('Output Heatmap')
    plt.axis('off')  # 不显示坐标轴

    plt.tight_layout()  # 调整子图间距
    plt.show()  # 显示图像和热力图

    xi,yi = top_centers[0]
    return xi,yi