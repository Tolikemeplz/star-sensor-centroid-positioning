import datetime
import os

import matplotlib
import torch

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import scipy.signal

import shutil
import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .utils import preprocess_input
from .utils_getlabel import get_label
from .utils_map import get_map


class LossHistory():
    def __init__(self, log_dir, model, input_shape):
        # 日志目录
        self.log_dir = log_dir
        # 存储训练损失
        self.losses = []
        # 存储验证损失
        self.val_loss = []
        # 存储验证偏差
        self.val_bias = []




        # 创建一个名为 log_dir 的目录，用于存储日志文件。
        os.makedirs(self.log_dir)
        # 创建一个 SummaryWriter 对象,用于记录和可视化训练过程中的数据。
        self.writer = SummaryWriter(self.log_dir)
        # try:
        #     dummy_input     = torch.randn(2, 3, input_shape[0], input_shape[1])
        #     self.writer.add_graph(model, dummy_input)
        # except:
        #     pass
    def append_bias(self,val_bias):
        self.val_bias.append(val_bias)


    def append_loss(self, epoch, loss, val_loss):
        """定义了一个名为 append_loss 的方法，用于向 losses 和 val_loss 列表中添加新的损失值。"""
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        self.val_loss.append(val_loss)

        # 将新的训练损失 loss 写入到名为 epoch_train_loss.txt 的文件中。
        with open(os.path.join(self.log_dir, "epoch_train_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        # 将新的验证损失 val_loss 写入到名为 epoch_val_loss.txt 的文件中。
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")



        # 使用 SummaryWriter 的 add_scalar 方法记录训练损失和验证损失。
        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        # 调用 loss_plot 方法，用于绘制损失曲线。
        self.loss_plot()

    def loss_plot(self):
        # 创建一个迭代器 iters，然后使用 matplotlib 绘制训练损失和验证损失曲线。
        iters = range(len(self.losses))
        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth=2, label='val loss')
        # 尝试对损失曲线进行平滑处理。如果失败，则忽略。
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15

            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle='--', linewidth=2,
                     label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle='--', linewidth=2,
                     label='smooth val loss')
        except:
            pass

        # 配置了损失曲线的图表：添加网格线、设置x轴和y轴的标签、以及图例的位置。
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        # 将绘制的损失曲线图保存为 epoch_loss.png 文件，存储在 log_dir 目录下。
        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        # 清除当前图形（cla 表示 clear axis），并关闭所有打开的图形窗口（close("all")）
        plt.cla()
        # # 创建一个新的图形来绘制验证偏差曲线
        # plt.figure()  # 创建第二个图形
        # plt.plot(iters, self.val_bias, 'blue', linewidth=2, label='val bias')
        # # 配置验证偏差曲线的图表：添加网格线、设置x轴和y轴的标签、以及图例的位置。
        # plt.grid(True)
        # plt.xlabel('Epoch')
        # plt.ylabel('Validation Bias')
        # plt.legend(loc="upper right")
        #
        # # 将绘制的验证偏差曲线图保存为 epoch_val_bias.png 文件，存储在 log_dir 目录下。
        # plt.savefig(os.path.join(self.log_dir, "epoch_val_bias.png"))
        # plt.cla()  # 清除当前图形

        plt.close("all")

class EvalCallback():
    def __init__(self, net, backbone, input_shape, val_lines, log_dir, cuda, gt_enlarge,\
            map_out_path=".temp_map_out", max_centers=4, confidence=0.05, MAXBIAS=0.5, eval_flag=True, period=1):
        self.net = net
        self.backbone = backbone
        self.input_shape = input_shape
        self.val_lines = val_lines
        self.log_dir = log_dir
        self.cuda = cuda
        self.map_out_path = map_out_path
        self.max_centers = max_centers
        self.confidence = confidence
        self.eval_flag = eval_flag
        self.period = period
        self.gt_enlarge = 2**gt_enlarge
        self.MAXBIAS = MAXBIAS

        self.aps = [0]
        self.bias= [0]
        self.epoches = [0]
        if self.eval_flag:
            with open(os.path.join(self.log_dir, "epoch_ap.txt"), 'a') as f:
                f.write(str(0))
                f.write("\n")
            with open(os.path.join(self.log_dir, "epoch_bias.txt"), 'a') as f:
                f.write(str(0))
                f.write("\n")


    # def get_map_txt(self, image_id, image, bias_out_path):
    #     f = open(os.path.join(bias_out_path, "detection-results/" + image_id + ".txt"), "w")
    #     # 计算输入图片的高和宽
    #     image_shape = np.array(np.shape(image)[0:2])
    #     # 因为输入的image是Image对象,只有2维
    #     image_np_expanded = np.expand_dims(image, axis=2)
    #
    #     #   图片预处理，归一化。获得的photo的shape为[1, 1, 128, 128]
    #     image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_np_expanded, dtype='float32')), (2, 0, 1)), 0)
    #
    #     with torch.no_grad():
    #         images = torch.from_numpy(np.asarray(image_data)).type(torch.FloatTensor)
    #         if self.cuda:
    #             images = images.cuda()
    #         #   将图像输入网络当中进行预测！
    #         outputs = self.net(images)
    #         #   利用预测结果进行解码
    #         outputs = get_label(outputs[0], outputs[1], self.gt_enlarge, self.confidence, self.cuda)
    #
    #         # 这里用result[0]可能是因为只传入了一张图片,所以只取第一张
    #         if outputs[0] is None:
    #             return
    #
    #         # 置信度
    #         top_conf = outputs[0][:, 2]
    #         # 预测星点坐标
    #         top_centers = outputs[0][:, :2]
    #
    #     # 根据置信度对预测框进行排序,并选择前self.max_boxes个最高的置信度框。
    #     top_100 = np.argsort(top_conf)[::-1][:self.max_centers]
    #     #  然后, 根据这个排序, 重新选择对应的边界狂,置信度和标签
    #     top_centers = top_centers[top_100]
    #     top_conf = top_conf[top_100]
    #
    #     for i, c in list(enumerate(top_conf)):
    #         centers = top_centers[i]
    #         score = str(top_conf[i])
    #
    #         xi, yi = centers
    #
    #         # 这表示变量score（假设是一个字符串）的前8个字符的切片
    #         f.write("%s %s %s\n" % (score[:8], str(xi), str(yi)))
    #
    #     f.close()
    #     return

    def get_map_txt(self, image_id, image, bias_out_path):
        f = open(os.path.join(bias_out_path, "detection-results/" + image_id + ".txt"), "w")
        # 计算输入图片的高和宽
        image_shape = np.array(np.shape(image)[0:2])
        # 因为输入的image是Image对象,只有2维
        image_np_expanded = np.expand_dims(image, axis=2)

        #   图片预处理，归一化。获得的photo的shape为[1, 1, 128, 128]
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_np_expanded, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(np.asarray(image_data)).type(torch.FloatTensor)
            if self.cuda:
                images = images.cuda()
            #   将图像输入网络当中进行预测！
            outputs = self.net(images)
            #   利用预测结果进行解码
            output = get_label(outputs[0], outputs[1], self.gt_enlarge, self.confidence, self.cuda)

            # 这里用result[0]可能是因为只传入了一张图片,所以只取第一张
            if output is None:
                return

            # 置信度
            top_conf = output[:, 2]
            # 预测星点坐标
            top_centers = output[:, :2]

        # 根据置信度对预测框进行排序,并选择前self.max_boxes个最高的置信度框。
        top_100 = np.argsort(top_conf)[::-1][:self.max_centers]
        #  然后, 根据这个排序, 重新选择对应的边界狂,置信度和标签
        top_centers = top_centers[top_100]
        top_conf = top_conf[top_100]

        for i, c in list(enumerate(top_conf)):
            centers = top_centers[i]
            score = str(top_conf[i])

            xi, yi = centers

            # 这表示变量score（假设是一个字符串）的前8个字符的切片
            f.write("%s %s %s\n" % (score[:8], str(xi), str(yi)))

        f.close()
        return



    def on_epoch_end(self, epoch, model_eval):
        # 初始化返回值为 'Not eval epoch'
        average_true_bias = 'Not eval epoch'
        # 检查当前epoch是否是self.period的倍数，并且self.eval_flag是否为True。
        if epoch % self.period == 0 and self.eval_flag:
            self.net = model_eval
            # 检查self.bias_out_path目录是否存在，如果不存在，则创建该目录。
            if not os.path.exists(self.map_out_path):
                os.makedirs(self.map_out_path)
            # 创建两个子目录"ground-truth"和"detection-results"，用于存储真实框和检测结果的文本文件。
            if not os.path.exists(os.path.join(self.map_out_path, "ground-truth")):
                os.makedirs(os.path.join(self.map_out_path, "ground-truth"))
            if not os.path.exists(os.path.join(self.map_out_path, "detection-results")):
                os.makedirs(os.path.join(self.map_out_path, "detection-results"))

            print("Get ground-truth and detection-results txt")
            for annotation_line in tqdm(self.val_lines):
                line = annotation_line.split()
                image_id = os.path.basename(line[0]).split('.')[0]
                image = Image.open(line[0]).convert('I')  # 假设图片是单通道的
                # 获得真实坐标
                gt_centers = np.array([np.array(list(map(float, box.split(',')))) for box in line[1:]])
                #  获得预测txt
                self.get_map_txt(image_id, image,  self.map_out_path)

                # 获得真实框txt
                with open(os.path.join(self.map_out_path, "ground-truth/" + image_id + ".txt"), "w") as new_f:
                    for center in gt_centers:
                        xi,yi,mv = center
                        new_f.write("%s %s\n" % (xi, yi))

            print("Calculate ap.")
            temp_ap,average_true_bias= get_map(self.MAXBIAS,False,path=self.map_out_path)
            self.aps.append(temp_ap)
            self.bias.append(average_true_bias)
            self.epoches.append(epoch)

            with open(os.path.join(self.log_dir, "epoch_ap.txt"), 'a') as f:
                f.write(str(temp_ap))
                f.write("\n")
            with open(os.path.join(self.log_dir, "epoch_bias.txt"), 'a') as f:
                f.write(str(average_true_bias))
                f.write("\n")

            # 创建并保存 self.maps 的图表
            plt.figure()  # 创建一个新的图表
            plt.plot(self.epoches, self.aps, 'red', linewidth=2, label='train ap')
            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Map %s' % str(self.MAXBIAS))
            plt.title('Map Curve')
            plt.legend(loc="upper right")
            plt.savefig(os.path.join(self.log_dir, "epoch_ap.png"))
            plt.cla()  # 清除当前图表

            # 创建并保存 self.bias 的图表
            plt.figure()  # 创建另一个新的图表
            plt.plot(self.epoches, self.bias, 'blue', linewidth=2, label='bias')
            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Bias %s' % str(self.MAXBIAS))
            plt.title('Bias Curve')
            plt.legend(loc="upper right")
            plt.savefig(os.path.join(self.log_dir, "epoch_bias.png"))
            plt.cla()  # 清除当前图表



            # 关闭所有打开的图表窗口
            plt.close("all")

            print("Get ap,bias and mv_bias done.")
            shutil.rmtree(self.map_out_path)
        return average_true_bias
        # else:
        #     return 'Not eval epoch'

















