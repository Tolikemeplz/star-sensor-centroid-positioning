import csv
import os
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom

def create_xml_from_csv(csv_path, images_folder, annotations_folder):
    # 确保输出文件夹存在
    if not os.path.exists(annotations_folder):
        os.makedirs(annotations_folder)

    # 读取CSV文件
    with open(csv_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # 跳过标题行
        for row in csvreader:
            image_name = row[0]
            x = row[1]
            y = row[2]
            mv = row[8]

            # 构建图片的完整路径
            image_path = os.path.join(images_folder, image_name)

            # 创建XML结构
            annotation = Element('annotation')
            folder = SubElement(annotation, 'folder')
            folder.text = 'images'
            filename = SubElement(annotation, 'filename')
            filename.text = image_name
            path = SubElement(annotation, 'path')
            path.text = image_path

            source = SubElement(annotation, 'source')
            database = SubElement(source, 'database')
            database.text = 'Unknown'

            size = SubElement(annotation, 'size')
            width = SubElement(size, 'width')
            width.text = '128'  # 假设所有图片都是128x128
            height = SubElement(size, 'height')
            height.text = '128'
            depth = SubElement(size, 'depth')
            depth.text = '1'

            segmented = SubElement(annotation, 'segmented')
            segmented.text = '0'

            # 添加目标点的坐标信息
            obj = SubElement(annotation, 'object')
            name = SubElement(obj, 'name')
            name.text = 'point'
            pose = SubElement(obj, 'pose')
            pose.text = 'Unspecified'
            truncated = SubElement(obj, 'truncated')
            truncated.text = '0'
            difficult = SubElement(obj, 'difficult')
            difficult.text = '0'
            star_mv = SubElement(obj, 'mv')
            star_mv.text = str(mv)
            point = SubElement(obj, 'point')
            x_coord = SubElement(point, 'x')
            x_coord.text = str(x)
            y_coord = SubElement(point, 'y')
            y_coord.text = str(y)

            # 将XML元素转换为字符串
            raw_xml = tostring(annotation, 'utf-8')
            parsed_xml = minidom.parseString(raw_xml)

            # 输出XML文件
            xml_path = os.path.join(annotations_folder, image_name.replace('.png', '.xml'))
            with open(xml_path, 'w') as xml_file:
                parsed_xml.writexml(xml_file, indent="  ")

    print("All XML files have been created.")

# 使用函数
csv_path = r'D:\programing\data\center-hybrid_dataset\csv\100.csv'
images_folder = r'D:\programing\data\center-hybrid_dataset\PNGImages'
annotations_folder = r'D:\programing\data\center-hybrid_dataset\Annotations\100'
create_xml_from_csv(csv_path, images_folder, annotations_folder)
