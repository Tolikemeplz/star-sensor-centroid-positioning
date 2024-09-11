import os
import random
import pandas
import xml.etree.ElementTree as ET

import numpy as np

#   annotation_mode用于指定该文件运行时计算的内容
#   annotation_mode为0代表整个标签处理过程，包括获得VOCdevkit/VOC2007/ImageSets里面的txt以及训练用的2007_train.txt、2007_val.txt
#   annotation_mode为1代表获得VOCdevkit/VOC2007/ImageSets里面的txt
#   annotation_mode为2代表获得训练用的2007_train.txt、2007_val.txt
annotation_mode     = 0

photo_nums  = np.zeros(2)

dataset_path = 'D:\programing\data\center-hybrid_dataset_200_small'
sets = ['train','val']


def convert_annotation(image_set,image_id, list_file):
    in_file = open(os.path.join(dataset_path, 'Annotations/%s/%s.xml' % (image_set, image_id)))
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        # difficult = 0
        # if obj.find('difficult') != None:
        #     difficult = obj.find('difficult').text
        # cls = obj.find('name').text

        point = obj.find('point')
        # mv为星等
        mv = obj.find('mv').text
        b = (float(point.find('x').text), float(point.find('y').text), float(mv))
        list_file.write(" " + ",".join([str(a) for a in b]))

        


if __name__ == "__main__":
    random.seed(0)
    if " " in os.path.abspath(dataset_path):
        raise ValueError("数据集存放的文件夹路径与图片名称中不可以存在空格，否则会影响正常的模型训练，请注意修改。")

    if annotation_mode == 0 or annotation_mode == 1:
        print("Generate txt in ImageSets.")
        # train_xmlfilepath存放了数据集里所有图片的xml文件,每个xml文件都存放了一张图片的对象的类别、位置和大小等信息。
        train_xmlfilepath = os.path.join(dataset_path, 'Annotations/train')
        # val_xmlfilepath存放了数据集里所有图片的xml文件,每个xml文件都存放了一张图片的对象的类别、位置和大小等信息。
        val_xmlfilepath = os.path.join(dataset_path, 'Annotations/val')
        # 定义saveBasePath，它是VOC2007数据集ImageSets/Main文件夹的路径，用于保存生成的文本文件。例如哪些图片被放在训练集,哪些被放在验证集了
        saveBasePath = os.path.join(dataset_path, 'ImageSets/Main')
        # 将所有train和val的xml文件名分别放在一个列表中
        train_temp_xml = os.listdir(train_xmlfilepath)
        val_temp_xml = os.listdir(val_xmlfilepath)
        train_total_xml = []
        for xml in train_temp_xml:
            if xml.endswith(".xml"):
                train_total_xml.append(xml)
        val_total_xml = []
        for xml in val_temp_xml:
            if xml.endswith(".xml"):
                val_total_xml.append(xml)

        # 训练集和验证集的数量
        tr = len(train_total_xml)
        vr = len(val_total_xml)
        print("train  size", tr)
        print("val  size", vr)

        # 打开两个文件，分别用于写入训练集和验证集的文件名。
        ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w')
        fval = open(os.path.join(saveBasePath, 'val.txt'), 'w')

        # 构造图片的文件名（去掉.xml扩展名）,并将其写入相应的文件中
        for i in range(tr):
            name = train_total_xml[i][:-4] + '\n'
            ftrain.write(name)
        for i in range(vr):
            name = val_total_xml[i][:-4] + '\n'
            fval.write(name)

        ftrain.close()
        fval.close()
        print("Generate txt in ImageSets done.")


    if annotation_mode == 0 or annotation_mode == 2:
        print("Generate dataset_train.txt and dataset_val.txt for train.")
        # 初始化type_index变量，用于跟踪当前处理的数据集类型（训练集或验证集）。
        type_index = 0

        for image_set in sets:
            # print(image_set)
            # 打开并读取ImageSets/Main文件夹中对应年份和图像集类型的文本文件（如train.txt或val.txt），将其内容分割成图像ID列表。
            image_ids = open(os.path.join(dataset_path, 'ImageSets/Main/%s.txt'% image_set), encoding='utf-8').read().strip().split()
            # 创建并打开一个新的文本文件，用于写入图像ID和对应的标注信息。
            list_file = open('dataset_200_small_%s.txt' % image_set, 'w', encoding='utf-8')
            # 遍历图像ID列表
            for image_id in image_ids:
                # print(f'imageid: {image_id}, imageset: {image_set}')
                # 将每个图像的完整路径写入到list_file中。
                list_file.write('%s/PNGImages/%s.png' % (os.path.abspath(dataset_path),  image_id))

                # 将这个图片对应的xml文件中的标注信息转换为文本格式,并写入list_file中
                convert_annotation(image_set, image_id, list_file)
                # 在每个图像的信息后添加一个换行符。
                list_file.write('\n')
            # 更新photo_nums数组，记录当前数据集类型的图像数量。例如train中有多少图片,val中有多少图片
            photo_nums[type_index] = len(image_ids)
            # 增加type_index，以便处理下一个数据集类型。
            type_index += 1
            list_file.close()
        print("Generate 2007_train.txt and 2007_val.txt for train done.")



















