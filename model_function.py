import numpy as np
import torch
import os
import torch.backends.cudnn as cudnn
from PIL import ImageDraw, ImageFont

from model.hybrid_centernet import Convnext_Centernet
from utils.utils import show_config,preprocess_input
from utils.utils_getlabel import get_label

class Model_function(object):
    _defaults = {
        # --------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
        #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        # --------------------------------------------------------------------------#
        "model_path": 'model_data/centernet_resnet50_voc.pth',

        # --------------------------------------------------------------------------#
        #   用于选择所使用的模型的主干
        #   resnet50, hourglass
        # --------------------------------------------------------------------------#
        "backbone": 'convnext',
        # --------------------------------------------------------------------------#
        #   输入图片的大小，设置成32的倍数
        # --------------------------------------------------------------------------#
        "input_shape": [128, 128],
        # --------------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        # --------------------------------------------------------------------------#
        "confidence": 0.3,
        # -------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        # -------------------------------#
        "cuda": True,

        "gt_enlarge":0,

        "phi":0,

        "max_centers":5



    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化model
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value



        self.generate()

        show_config(**self._defaults)

    def generate(self, onnx=False):
        #-------------------------------#
        #   载入模型与权值
        #-------------------------------#
        assert self.backbone in ['convnext']
        if self.backbone == "convnext":
            self.net = Convnext_Centernet(bifpn_repeat=1, pretrained=False, gt_enlarge=self.gt_enlarge, phi=self.phi)
        else:
            self.net = Convnext_Centernet({'hm': self.num_classes, 'wh': 2, 'reg':2})

        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = torch.nn.DataParallel(self.net)
                self.net = self.net.cuda()


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
            output = get_label(outputs[0], outputs[1], 2**self.gt_enlarge, self.confidence, self.cuda)

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



