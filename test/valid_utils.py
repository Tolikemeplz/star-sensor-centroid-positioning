import glob
import json
import math
import operator
import os
import ast
import shutil
import pandas as pd
import sys
import cv2
import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def error(msg):
    print(msg)
    sys.exit(0)

def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0) # insert 0.0 at begining of list
    rec.append(1.0) # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0) # insert 0.0 at begining of list
    prec.append(0.0) # insert 0.0 at end of list
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab: for i=numel(mpre)-1:-1:1
                    mpre(i)=max(mpre(i),mpre(i+1));
    """
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    """
     This part creates a list of indexes where the recall changes
        matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i) # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre

def file_lines_to_list(path):
    # open txt file lines to a list
    with open(path) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content

def get_dic(MAXBIAS,RESULT_PATH,CSV_PATH,score_threhold=0.5 , path = './map_out',is_apart=False,apart_mid=None):
    GT_PATH = os.path.join(path, 'ground-truth')
    DR_PATH = os.path.join(path, 'detection-results')
    TEMP_FILES_PATH = os.path.join(path, '.temp_files')
    TEMP_FILES_PATH_gt_json =os.path.join(TEMP_FILES_PATH, '.gt_json')

    # 如果临时文件路径TEMP_FILES_PATH不存在，则创建它。
    if not os.path.exists(TEMP_FILES_PATH):
        os.makedirs(TEMP_FILES_PATH)

        # 如果临时文件路径TEMP_FILES_PATH不存在，则创建它。
        if not os.path.exists(TEMP_FILES_PATH_gt_json):
            os.makedirs(TEMP_FILES_PATH_gt_json)

    # 使用glob模块来查找并返回一个列表，其中包含特定目录（由GT_PATH指定）下所有.txt文件的路径。
    ground_truth_files_list = glob.glob(GT_PATH + '/*.txt')
    if len(ground_truth_files_list) == 0:
        error("Error: No ground-truth files found!")
    # 将ground_truth_files_list列表中的文件路径进行排序。
    ground_truth_files_list.sort()
    # 用于跟踪地面真实数据的个数
    gt_counter = 0

    for txt_file in ground_truth_files_list:
        # 对于每个文件路径，这行代码通过分割文件名来获取不带扩展名的文件标识符
        # 即当前图片的图片名
        file_id = txt_file.split(".txt", 1)[0]
        # 使用os.path.basename获取路径的最后一部分，并使用os.path.normpath规范化路径，以确保路径在不同操作系统中的兼容性。
        file_id = os.path.basename(os.path.normpath(file_id))
        # 使用file_id和检测结果的路径DR_PATH来构建检测结果的临时路径temp_path。
        temp_path = os.path.join(DR_PATH, (file_id + ".txt"))
        # 如果检测结果的临时文件不存在，则调用error函数来报告错误。
        if not os.path.exists(temp_path):
            error_msg = "Error. File not found: {}\n".format(temp_path)
            error(error_msg)
        # 读取地面真实标注文件的所有行，并将其转换为列表。
        lines_list = file_lines_to_list(txt_file)
        # 初始化一个空列表centers，用于存储当前文件的所有星点坐标信息。
        centers = []
        # 初始化一个标志变量is_difficult，用于标记当前边界框是否属于困难样本。
        is_difficult = False
        # lines_list是图片中的所有center,每个line对应一个center
        for line in lines_list:
            try:
                # 尝试使用空格分割行。如果行包含“difficult”标记，则期望有3个字段（xi,yi和“difficult”标记）。
                if "difficult" in line:
                    xi, yi, _difficult = line.split()
                    is_difficult = True
                # 否则，期望有2个字段（xi,yi）。但是，这种方法假设每个字段之间只有一个空格，并且字段不包含空格。
                else:
                    xi, yi, mv= line.split()
            # 如果split()方法抛出异常（例如，由于列数不匹配），将执行except块中的代码。
            except:
                # 对于包含"difficult"的行,这些行被假定为具有不规则的格式，因此代码尝试通过列表切片和遍历来手动解析它们。
                if "difficult" in line:
                    line_split = line.split()
                    _difficult = line_split[-1]
                    mv = line_split[-2]
                    yi = line_split[-3]
                    xi = line_split[-4]
                    is_difficult = True
                # 对于不包含"difficult"的行,类似于上面的逻辑，但由于缺少"difficult"标记，解析的逻辑略有不同。
                else:
                    line_split = line.split()
                    mv = line_split[-1]
                    yi = line_split[-2]
                    xi = line_split[-3]

            # 将解析得到的星点坐标（xi,yi）组合成一个字符串
            center = xi + " " + yi
            # 如果是困难样本，将一个字典添加到bounding_boxes列表中。
            if is_difficult:
                # 包含边界框的类别名称（class_name）、边界框坐标字符串（bbox）、
                # 一个标记边界框是否被使用的布尔值（used，初始为False），以及一个表示边界框是否是困难样本的布尔值
                centers.append({ "center": center, "used": False, "difficult": True, "mv":float(mv)})
                is_difficult = False
            else:
                # 如果边界框不是困难样本，则同样将其添加到bounding_boxes列表中，但这次不包括“difficult”标志。
                centers.append({"center":center, "used":False, "mv":float(mv)})
                gt_counter += 1

        # 最后，使用json.dump函数将centers列表写入一个JSON文件，文件名基于file_id。
        # 这样，每个地面真实文件的内容都被转换为一个JSON文件，方便后续处理。
        # 也就是说每个图片对应一个JSON文件
        with open(TEMP_FILES_PATH_gt_json + "/" + file_id + "_ground_truth.json", 'w') as outfile:
            json.dump(centers, outfile)


    # 这行代码使用glob模块的glob函数来查找DR_PATH目录下所有以.txt结尾的文件。
    # 这些文件假定是检测结果文件，包含了检测算法的输出。
    dr_files_list = glob.glob(DR_PATH + '/*.txt')
    # 这行代码对获取的检测结果文件列表dr_files_list进行排序
    dr_files_list.sort()

    # 初始化一个空的列表centers来存储匹配的星点坐标信息。注意这里是所有图片的所有星点
    dr_centers = []
    # 遍历之前获取的检测结果文件列表dr_files_list。
    # 每个txt_file都是一张图片
    for txt_file in dr_files_list:
        # 从当前检测结果文件路径中提取文件名的基本部分，去除扩展名
        file_id = txt_file.split(".txt", 1)[0]
        # 进一步处理file_id，使用os.path.basename获取规范化后的路径的最后一部分，确保文件名的一致性。
        file_id = os.path.basename(os.path.normpath(file_id))
        # 使用这个文件ID和地面真实数据的路径GT_PATH来构建对应的地面真实文件路径temp_path
        temp_path = os.path.join(GT_PATH, (file_id + ".txt"))
        # 检查地面真实文件是否存在
        if not os.path.exists(temp_path):
            error_msg = "Error. File not found: {}\n".format(temp_path)
            error(error_msg)
        # 使用file_lines_to_list函数（该函数在这段代码中没有给出定义）读取文件的每一行，并遍历这些行。
        lines = file_lines_to_list(txt_file)
        # lines是这张图片中检测出的所有center,line是其中一个center
        for line in lines:
            # 代码尝试使用空格分割每一行，并期望得到置信度、星点坐标和星等。
            # 但是，如果行格式不符合预期（例如，因为类别名包含空格），则代码会捕获异常并尝试另一种解析方法。
            try:
                confidence, xi, yi = line.split()
            except:
                line_split = line.split()
                yi = line_split[-1]
                xi = line_split[-2]
                confidence = line_split[-3]

            center = xi + " " + yi
            dr_centers.append({"confidence": confidence, "file_id": file_id, "center": center})


    # 这行代码对dr_centers列表进行就地排序。列表中的每个元素都是一个包含检测结果的字典。
    # key参数指定了排序的依据，这里使用了lambda函数来获取每个字典中confidence键对应的值，并将其转换为浮点数。
    # reverse=True参数指定了排序的顺序为降序，即置信度最高的检测结果排在列表的前面。
    dr_centers.sort(key=lambda x:float(x['confidence']), reverse=True)
    # 这行代码使用with语句打开一个文件，文件路径由TEMP_FILES_PATH和扩展名_dr.json组成。
    with open(TEMP_FILES_PATH + "/" +  "_dr.json", 'w') as outfile:
        # json.dump函数将dr_centers列表转换为JSON格式，并将其写入打开的文件对象outfile中。
        json.dump(dr_centers, outfile)

    # 这行代码初始化一个空变量count_true_positives。这个字典将用于存储真正例（true positives）的计数。
    # 在物体检测任务中，真正例通常指的是模型正确检测到的对象实例。初始化真正例（TP）计数为0。
    count_true_positives = 0
    # 存储所有正确检测的星的位置偏差
    true_positives_bias = []

    # 这段代码是评估目标检测算法性能的核心部分，具体用于计算平均精度（AP）、F1分数、召回率和精度，
    # 并可能展示一个可视化的动画来展示检测结果与地面真实数据的匹配情况。
    # 加载该类别对应的检测结果JSON文件,并将其内容解析为JSON对象。
    dr_file = TEMP_FILES_PATH + "/" + "_dr.json"
    dr_data = json.load(open(dr_file))

    # 获取检测结果的数量。
    nd = len(dr_data)
    # 初始化检测数量（nd）、真正例（tp）、假正例（fp）和置信度（score）列表。
    tp = [0] * nd
    fp = [0] * nd
    score = [0] * nd
    # 初始化一个变量，用于记录得分阈值索引。
    score_threhold_idx = 0

    # 遍历检测结果，与地面真实数据进行比较，计算相关指标（如真正例、假正例），并可能显示一个可视化动画。
    # dr_data是所有center,idx, detection是其中的一个center
    for idx, detection in enumerate(dr_data):
        # 从当前检测结果中提取文件ID和置信度，并将置信度存储到列表score中
        file_id = detection["file_id"]
        score[idx] = float(detection["confidence"])
        # 如果当前检测结果的置信度高于或等于设定的阈值score_threhold，则更新score_threhold_idx为当前索引。
        # 因为dr_data是按照置信度排序过,所以当循环完一遍以后,排在idx=score_threhold_idx之前的box就是置信度大于score_threhold的box
        if score[idx] >= score_threhold:
            score_threhold_idx = idx
        # 这里首先初始化了一个长度为 nd（检测结果数量）的列表 score，用于存储每个检测结果的置信度分数。然后遍历检测结果列表 dr_data，将每个检测结果的置信度分数存入 score 列表的对应位置。如果某个检测结果的置信度分数大于或等于 score_threhold，则更新 score_threhold_idx 为该检测结果在列表中的索引。
        # 之后，在计算召回率、精度等指标时，可以使用 score_threhold_idx 来确定哪些检测结果应该被考虑在内。例如，在计算F1分数、召回率和精度时，可能只会考虑 score_threhold_idx 及其之前的检测结果。

        # 使用文件ID加载对应的地面真实数据（JSON格式）,即这个box对应的gt图片
        gt_file = TEMP_FILES_PATH_gt_json + "/" + file_id + "_ground_truth.json"
        ground_truth_data = json.load(open(gt_file))

        # 初始化一个变量bias_min，用于存储当前检测结果的最小平均偏差
        biasmin = 128
        # 初始化一个变量gt_match，用于存储与当前检测结果最佳匹配的地面真实box
        gt_match = -1
        # 将当前检测结果的坐标从字符串转换为浮点数列表。
        cc = [float(x) for x in detection["center"].split()]
        # 这个循环遍历这个检测的center对应的地面真实数据图片中的所有的真实星点的坐标
        for obj in ground_truth_data:
            # 将当前地面真实对象的星点坐标从字符串格式分割并转换成浮点数列表ccgt。
            ccgt = [float(x) for x in obj["center"].split()]
            # 计算当前检测结果的边界框（cc）和当前地面真实对象的边界框（ccgt）的偏差
            # 计算x轴和y轴上的距离
            distance_x = abs(cc[0] - ccgt[0])
            distance_y = abs(cc[1] - ccgt[1])
            # 计算L1误差
            bias = (distance_x + distance_y)
            if bias < biasmin:
                biasmin = bias
                # 将当前地面真实对象的数据赋值给gt_match，表示这个对象是与当前检测结果匹配度最高的地面真实对象。
                gt_match = obj

        max_bias = MAXBIAS
        # 如果预测坐标和真实星点坐标之间的最小偏差小于或等于max_bias
        if biasmin <= max_bias:
            # 如果这个真星点坐标还没有被使用过（即还没有与之匹配的预测坐标）
            if not bool(gt_match["used"]):
                # 将当前索引idx对应的TP设置为1，表示这是一个真正例
                tp[idx] = 1
                # 将这个真实框标记为已使用
                gt_match["used"] = True
                # 将修改后的ground_truth_data数据写回到由gt_file指定的JSON文件中，覆盖该文件原有的内容（如果存在的话）。
                # 这通常用于在评估目标检测算法时，标记哪些地面真实对象已经被匹配到，以便后续处理。
                # 因为修改了gt_match就是修改了obj,就是修改了ground_truth_data
                gt_match["bias"] = biasmin
                # 计算检测星等和分配的真实星的星等的差值
                count_true_positives += 1
                true_positives_bias.append(biasmin)
                with open(gt_file, 'w') as f:
                    f.write(json.dumps(ground_truth_data))
            # 因为dr_data是按照置信度排过序的,所以如果一个真实框被一个置信度更高的
            # 预测框匹配上了,不管之后匹配上这个真实框的置信度更小的预测框的ovmax是多少,都认为是假正例,因为以置信度优先
            else:
                fp[idx] = 1
        else:
            # 将fp列表中当前检测结果对应的索引idx位置的值设置为1，表示这是一个假正例
            # 因为其实只要是检测出来的预测框其实都相当于是正例.
            fp[idx] = 1

    print(f'共有{count_true_positives}个正例')

    cumsum = 0  # 初始化一个变量cumsum为0，用于跟踪累积的FP或TP值。

    # 这段代码的目的是对 fp 列表中的每个元素进行累积求和，使得 fp[i] 变成前 i+1 个检测结果的假正例总数。
    # fp是这一类的预测框是否是假正例的一个标记列表.这样做的好处就是,fp[score_threhold_idx]的值刚好就是有多少个置信度满足要求的真正例的值
    for idx, val in enumerate(fp):
        fp[idx] += cumsum
        cumsum += val

    # 这段代码试图对真正例计数进行累积求和。
    cumsum = 0
    for idx, val in enumerate(tp):
        tp[idx] += cumsum
        cumsum += val

    # 创建tp列表的一个副本，命名为rec，用于后续计算召回率。
    rec = tp[:]
    for idx, val in enumerate(tp):  # 这里，tp 是真正例计数列表,gt_counter 是地面真实数据中的对象数量。
        # 召回率是真正例数量除以地面真实对象总数。使用 np.maximum(gt_counter, 1) 是为了确保分母不为零
        rec[idx] = float(tp[idx]) / np.maximum(gt_counter, 1)

    # 精度是真正例数量除以（真正例数量加假正例数量）。
    prec = tp[:]
    for idx, val in enumerate(tp):
        # 这里，我们使用累积的假正例计数
        prec[idx] = float(tp[idx]) / np.maximum((fp[idx] + tp[idx]), 1)

    # 这行代码调用一个名为voc_ap的函数，传入召回率（rec）和精确度（prec）的数组。这个函数可能是根据PASCAL VOC数据集的评估标准来计算平均精度的。函数返回三个值：平均精度（ap），修改后的召回率数组（mrec），和修改过的精确度数组（mprec）。
    ap, mrec, mprec = voc_ap(rec[:], prec[:])
    # 这行代码计算F1分数。
    F1 = np.array(rec) * np.array(prec) * 2 / np.where((np.array(prec) + np.array(rec)) == 0, 1,
                                                       (np.array(prec) + np.array(rec)))

    # 格式化一个字符串，显示 AP值和类别名称。这里使用了字符串格式化来将 AP 值转换为百分比形式，并附加到类别名称后面。
    text = "{0:.2f}%".format(ap * 100) + " = " + " AP "  # " AP = {0:.2f}%".format(ap*100)

    # 因为fp和tp的累加行为,并且rec又是除以过gt_counter_per_class[class_name]的,而又因为fp和tp是按照置信度来排序的,所以
    # F1[score_threhold_idx]其实就是置信度满足要求,即大于score_threhold=0.5的所有这一类的预测框的平均结果,可以代表这个类别的F1(不一定正确),rec[score_threhold_idx]等同理
    if len(prec) > 0:
        F1_text = "{0:.5f}".format(F1[score_threhold_idx]) + " = " + " F1 "
        Recall_text = "{0:.5f}%".format(rec[score_threhold_idx] * 100) + " = " + " Recall "
        Precision_text = "{0:.5f}%".format(prec[score_threhold_idx] * 100) + " = " + " Precision "
        Bias_text = "{0:.5f}".format(sum(true_positives_bias)/(count_true_positives+1e-12)) + " = " + " Bias "
    else:
        F1_text = "0.00" + " = " + " F1 "
        Recall_text = "0.00%" + " = " + " Recall "
        Precision_text = "0.00%" + " = " + " Precision "
        Bias_text = "0.00" + " = " + " Bias "
    with open(RESULT_PATH + "/" + 'result.txt', 'w') as file:
        # 写入 AP 值
        file.write(text + "\n")  # 添加换行符以便区分不同行的内容

        # 如果 prec 列表不为空，则写入 F1、Recall 和 Precision 值
        if len(prec) > 0:
            file.write(F1_text + "\n")
            file.write(Recall_text + "\n")
            file.write(Precision_text + "\n")
            file.write(Bias_text + "\n")
        else:
            file.write(F1_text + "\n")
            file.write(Recall_text + "\n")
            file.write(Precision_text + "\n")
            file.write(Bias_text + "\n")

    gt_centers = []
    # 使用glob模块来查找并返回一个列表，其中包含特定目录（由TEMP_FILES_PATH_gt_json指定）下所有.json文件的路径。
    ground_truth_json_list = glob.glob(TEMP_FILES_PATH_gt_json + '/*.json')
    # 读取CSV文件
    df = pd.read_csv(CSV_PATH, low_memory=False)
    for json_file in ground_truth_json_list:
        # file_id = json_filename.split('_')[0]
        # 即当前图片的图片名
        file_id = json_file.split(".json", 1)[0]
        # 使用os.path.basename获取路径的最后一部分，并使用os.path.normpath规范化路径，以确保路径在不同操作系统中的兼容性。
        file_id = os.path.basename(os.path.normpath(file_id))
        file_id = file_id.split('_ground_truth')[0]
        # print('file_id:',file_id)
        ground_truth_data = json.load(open(json_file))
        for obj in ground_truth_data:
            obj["file_id"] = file_id
            #df = pd.read_csv(CSV_PATH)
            # 构建正则表达式来匹配name列的格式
            # import re
            # pattern = re.compile(f'^{file_id}\\\.png$')
            # # 使用正则表达式筛选出包含特定file_id的行
            # matching_rows = df[df['name'].str.match(pattern)]
            matching_rows = df[df['name'] == file_id + '.png']
            # 假设你想获取这一行的其他列，例如'column1'和'column2'
            if not matching_rows.empty:
                gauss_p_str = matching_rows.iloc[0]['gauss_p']
                # 将字符串转换为字典
                gauss_p = ast.literal_eval(gauss_p_str)
                obj["gauss_mean"] = gauss_p["mean"]
                obj["gauss_variance"] = gauss_p["variance"]
                poisson_p_str = matching_rows.iloc[0]["poisson_p"]
                poisson_p = ast.literal_eval(poisson_p_str)
                obj["dark_noise"] = poisson_p["dark_noise"]
                obj["e"] = poisson_p["e"]

                gt_centers.append(obj)
        with open(json_file, 'w') as f:
            f.write(json.dumps(ground_truth_data))
    with open(RESULT_PATH + "/" + "_gt.json", 'w') as outfile:
        # json.dump函数将dr_centers列表转换为JSON格式，并将其写入打开的文件对象outfile中。
        json.dump(gt_centers, outfile)

    if is_apart is True:
        return ap,F1[score_threhold_idx],rec[score_threhold_idx],prec[score_threhold_idx],sum(true_positives_bias)/(count_true_positives+1e-12)



def add_dic(RESULT_PATH,CSV_PATH,pic_type):
    gt_file = RESULT_PATH + "/" + "_gt.json"
    # 使用with语句确保文件正确关闭
    with open(gt_file, 'r') as file:
        gt_data = json.load(file)

    # 读取CSV文件
    df = pd.read_csv(CSV_PATH, low_memory=False)
    for idx, detection in enumerate(gt_data):

        # 构建正则表达式来匹配name列的格式
        import re
        # pattern = re.compile(f'^{detection["file_id"]}\\\.png$')
        # # 使用正则表达式筛选出包含特定file_id的行
        # matching_rows = df[df['name'].str.match(pattern)]
        matching_rows = df[df['name'] == detection["file_id"] + '.png']
        # 假设你想获取这一行的其他列，例如'column1'和'column2'
        if not matching_rows.empty:
            if pic_type == 'angular':
                xtyt_p_str = matching_rows.iloc[0]['xtyt_p']
                xtyt_p = ast.literal_eval(xtyt_p_str)
                detection["wxy"] = np.sqrt(xtyt_p["wx"]**2 +  xtyt_p["wy"]**2)
                detection["wz"] = np.sqrt(xtyt_p["wz"]**2)

    with open(RESULT_PATH + "/" + "_gt.json", 'w') as outfile:
        # json.dump函数将dr_centers列表转换为JSON格式，并将其写入打开的文件对象outfile中。
        json.dump(gt_data, outfile)









def bias_plot(factor,RESULT_PATH,part=100,smooth=False):
    gt_file = RESULT_PATH + "/" + "_gt.json"
    # 使用with语句确保文件正确关闭
    with open(gt_file, 'r') as file:
        gt_data = json.load(file)

    factor_data=[]
    bias=[]

    for idx, detection in enumerate(gt_data):
        if detection["used"] == True:
            factor_data.append((detection[f"{factor}"],detection["bias"]))
            bias.append(detection["bias"])

    # 对factor_data列表进行排序，依据每个元组中的第一个元素（factor值）
    factor_data_sorted = sorted(factor_data, key=lambda item: item[0])
    # 分离排序后的factor值和bias值
    factors, biases = zip(*factor_data_sorted)

    # 创建图表
    plt.figure(figsize=(20, 4))
    # 绘制图表，使用线图和散点图
    plt.plot(factors, biases, marker='o', linestyle='', color='b', label=f'Bias vs. {factor}')

    # 在图表上显示指定的十个位置的bias值
    num_points = 10
    for i in range(num_points):
        # 计算索引位置
        index = int(i * (len(factors) - 1) / (num_points - 1))
        # 在图表上显示文本
        plt.text(factors[index], biases[index], f'{biases[index]:.2f}', fontsize=9, ha='center')
    # 添加图例
    plt.legend()

    # 设置图表标题和坐标轴标签
    plt.title(f'Relationship Between {factor} and Bias')
    plt.xlabel(f'{factor} Value')
    plt.ylabel('Bias Value')
    # 显示网格
    plt.grid(True)

    # 获取RESULT_PATH的值，并确保路径以'/'结尾
    if not RESULT_PATH.endswith('/'):
        RESULT_PATH += '/'
    # 图表文件名
    chart_filename = f'bias_vs_{factor}_chart.png'
    # 完整的文件路径
    chart_path = RESULT_PATH + chart_filename

    # 保存图表到指定路径
    plt.savefig(chart_path)
    # 显示图表
    # plt.show()
    print(f'图表已保存至 {chart_path}')

    # 将factor从小到大均匀分成100个区间
    min_factor, max_factor = min(factors), max(factors)
    intervals = np.linspace(min_factor, max_factor, part+1)
    mid_points = 0.5 * (intervals[:-1] + intervals[1:])
    mean_biases = []

    # 计算每个区间内的bias平均值
    for i in range(part):
        lower_bound = intervals[i]
        upper_bound = intervals[i + 1]
        biases_in_interval = [b for f, b in factor_data_sorted if lower_bound <= f < upper_bound]
        mean_bias = np.mean(biases_in_interval) if biases_in_interval else 0
        mean_biases.append(mean_bias)

    if smooth:
        # 使用二次多项式拟合数据
        coefficients = np.polyfit(mid_points, mean_biases, 2)  # 2代表二次多项式
        polynomial = np.poly1d(coefficients)
        smooth_curve = polynomial(mid_points)
    else:
        smooth_curve = mean_biases

    # 创建图表
    plt.figure(figsize=(20, 4))
    plt.plot(mid_points, smooth_curve, marker='o', linestyle='-', color='b', label=f'Average Bias vs. {factor}')

    # 设置图表标题和坐标轴标签
    plt.title(f'Relationship Between {factor} and Average Bias')
    plt.xlabel(f'{factor} Mid Point Value')
    plt.ylabel('Average Bias Value')
    plt.grid(True)
    plt.legend()

    # 确保RESULT_PATH以'/'结尾
    if not RESULT_PATH.endswith('/'):
        RESULT_PATH += '/'
    chart_filename = f'avg_bias_vs_{factor}_chart.png'
    chart_path = RESULT_PATH + chart_filename

    # 保存图表
    plt.savefig(chart_path)
    # plt.show()
    print(f'图表已保存至 {chart_path}')


# 定义一个函数来绘制图表并保存
def plot_and_save(groups, title, filename, result_path):
    x, y = zip(*groups[:-1])  # 解包元组列表为两个列表
    plt.figure(figsize=(10, 6))  # 设置图表大小
    plt.plot(x, y, marker='o')  # 绘制线条，数据点标记为圆圈
    plt.title(title)  # 设置图表标题
    plt.xlabel('Factor')  # 设置横坐标标签
    plt.ylabel('Score')  # 设置纵坐标标签
    plt.savefig(os.path.join(result_path, filename))  # 保存图表
    plt.close()  # 关闭图表以释放资源


def plot_and_save_objects(groups1, groups2, groups3,x_label, y_label, filename, result_path, removelast=0):
    plt.figure(figsize=(7, 5))  # 调整图表大小以适应论文版面

    # 设置图表样式和字体
    plt.style.use('default')
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.facecolor'] = 'white'  # 设置图表背景为白色
    plt.rcParams['axes.edgecolor'] = 'black'  # 设置坐标轴颜色为黑色
    plt.rcParams['axes.grid'] = False  # 禁用网格

    # 根据removelast参数处理groups1
    if removelast > 0:
        x1, y1 = zip(*groups1[:-removelast])
    else:
        x1, y1 = zip(*groups1)
    plt.plot(x1, [y*100 for y in y1], marker='o', linestyle='-', linewidth=1.5, markersize=2, label='rec')  # 绘制线条和数据点

    # 处理groups2
    if removelast > 0:
        x2, y2 = zip(*groups2[:-removelast])
    else:
        x2, y2 = zip(*groups2)
    plt.plot(x2, [y*100 for y in y2], marker='s', linestyle='-', linewidth=1.5, markersize=2, label='prec')  # 绘制线条和数据点

    # 处理groups3
    if removelast > 0:
        x3, y3 = zip(*groups3[:-removelast])
    else:
        x3, y3 = zip(*groups3)
    plt.plot(x3, [y*100 for y in y3], marker='^', linestyle='-', linewidth=1.5, markersize=2, label='F1')  # 绘制线条和数据点

    # 设置y轴的最大值
    plt.ylim(97, 100)
    #plt.title(title, fontsize=12)  # 设置图表标题和字体大小
    plt.xlabel(x_label, fontsize=10)  # 设置横坐标标签和字体大小
    plt.ylabel(y_label, fontsize=10)  # 设置纵坐标标签和字体大小
    # 设置y轴为百分比格式
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:.0f}%'.format(y)))

    plt.xticks(fontsize=10)  # 设置 x 轴刻度字体大小
    plt.yticks(fontsize=10)  # 设置 y 轴刻度字体大小
    plt.legend(fontsize=8)  # 显示图例

    # 移除辅助虚线
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.tight_layout()  # 调整布局以适应图表内容
    plt.savefig(os.path.join(result_path, filename))
    plt.close()  # 关闭图表以释放资源

def plot_bar_and_save(groups, title, filename, result_path):
    x, y = zip(*groups)  # 解包元组列表为两个列表
    plt.figure(figsize=(10, 6))  # 设置图表大小
    plt.bar(x, y)  # 绘制条形图
    plt.title(title)  # 设置图表标题
    plt.xlabel('Factor')  # 设置横坐标标签
    plt.ylabel('Score')  # 设置纵坐标标签
    plt.xticks(rotation=45)  # 旋转x轴标签，提高可读性
    plt.tight_layout()  # 自动调整子图参数，使之填充整个图表区域
    plt.savefig(os.path.join(result_path, filename))  # 保存图表
    plt.close()  # 关闭图表以释放资源























