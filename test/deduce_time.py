import time
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
from model.hybrid_centernet import Convnext_Centernet
from utils.utils import preprocess_input

model =Convnext_Centernet(bifpn_repeat=1,pretrained=False,gt_enlarge=0,phi=8)
# 确保模型在正确的设备上
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # 设置为评估模式
image_path = r'self_image\1.png'


image = Image.open(image_path)
image_array = np.array(image)
image_np_expanded = np.expand_dims(image_array, axis=2)
image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_np_expanded, dtype='float32')), (2, 0, 1)),0)
images = torch.from_numpy(np.asarray(image_data)).type(torch.FloatTensor)

# # 测试CPU推理时间
# with torch.no_grad():
#     model.cpu()  # 确保模型在CPU上
#     start_time = time.time()
#     output = model(images)
#     cpu_time = time.time() - start_time
#     print(f"CPU推理时间: {cpu_time:.4f}秒")
#
# # 测试GPU推理时间
# if torch.cuda.is_available():
#     with torch.no_grad():
#         model.to(device)  # 确保模型在GPU上
#         img_t = images.to(device)
#         start_time = time.time()
#         output = model(img_t)
#         gpu_time = time.time() - start_time
#         print(f"GPU推理时间: {gpu_time:.4f}秒")
# else:
#     print("CUDA不可用，无法测试GPU推理时间。")

# 定义一个函数来测量推理时间
def measure_inference_time(model, images, device, num_warmups=20, num_inferences=200):
    model.to(device)
    images = images.to(device)
    with torch.no_grad():
        # 热身阶段，不计算时间
        for _ in range(num_warmups):
            _ = model(images)

        # 测量时间
        start_time = time.time()
        for _ in range(num_inferences):
            _ = model(images)
        end_time = time.time()

        # 计算平均推理时间
        average_time = (end_time - start_time) / num_inferences
        return average_time


# 测试CPU推理时间
cpu_time = measure_inference_time(model, images, torch.device('cpu'))
print(f"CPU推理时间（去掉前10次后的平均时间）: {cpu_time:.4f}秒")

# 测试GPU推理时间
if torch.cuda.is_available():
    gpu_time = measure_inference_time(model, images, torch.device('cuda'))
    print(f"GPU推理时间（去掉前10次后的平均时间）: {gpu_time:.4f}秒")
else:
    print("CUDA不可用，无法测试GPU推理时间。")