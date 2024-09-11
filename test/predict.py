import torch
import time
import os
import numpy as np
from PIL import Image
from predict_utils import get_map_txt
from torchvision import transforms
from model.hybrid_centernet import Convnext_Centernet, Convnext_Centernet_e, Convnext_Centernet_small
from utils.utils import preprocess_input



def predict(image_path,weight_path,bias_out_path,confidence,max_centers,net='convnext',phi=8):
    # 确保图像和模型都在同一个设备上
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    image = Image.open(image_path).convert('I')  # 假设图片是单通道的
    # 将PIL图像转换为numpy数组以获取通道数
    image_array = np.array(image)
    # 获取图像的形状
    shape = image_array.shape
    # print(f"图像的形状是: {shape}")

    image_id = os.path.splitext(os.path.basename(image_path))[0]
    if net == 'convnext':
        net = Convnext_Centernet(bifpn_repeat=1,pretrained=False,gt_enlarge=0,phi=phi)
    elif net == 'efficent':
        net = Convnext_Centernet_e(bifpn_repeat=1, pretrained=False,gt_enlarge=0,phi=phi,bx=1)
    elif net == 'convnext_small':
        net = Convnext_Centernet_small(bifpn_repeat=1, gt_enlarge=0, pretrained=False,phi=8)
    net = net.to(device)  # 将模型移到指定设备
    weight = torch.load(weight_path)
    net.load_state_dict(weight)
    xi,yi = get_map_txt(image_id, net, image, bias_out_path, confidence, max_centers,gt_enlarge=0)
    return xi,yi


if __name__ == "__main__":
    image_id = '3_acde_bezier_both_33922'
    weight = 'best_loss_epoch_weights.pth'
    confidence = 0.1
    max_centers = 5

    #image_root_path = r'D:\！！工作学习\理化所\！！课题\高动态星敏\信息收集\lozhiwen'
    image_root_path = r'D:\programing\research_code\centernet-hybrid-withoutmv-predict\test\self_image'
    weight_root_path = r'D:\programing\data\center-hybrid_dataset_200\train_result\before_withoutmv-dropout-1'
    bias_out_path = r'D:\programing\research_code\centernet-hybrid-withoutmv-predict\test\predict_result'
    image_path = os.path.join(image_root_path, image_id + ".png")
    weight_path = os.path.join(weight_root_path, weight)

    xi,yi = predict(image_path,weight_path,bias_out_path,confidence,max_centers)






