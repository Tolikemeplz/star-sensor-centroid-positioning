from matplotlib import pyplot as plt
from replicate.used import generate_uniform_distribution
import matplotlib.ticker as ticker
import csv
import os
import numpy as np

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
    plt.ylim(97, 100.1)
    #plt.title(title, fontsize=12)  # 设置图表标题和字体大小
    plt.xlabel(x_label, fontsize=12)  # 设置横坐标标签和字体大小
    plt.ylabel(y_label, fontsize=12)  # 设置纵坐标标签和字体大小
    # 设置y轴为百分比格式
    #plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:.0f}%'.format(y)))

    plt.xticks(fontsize=12)  # 设置 x 轴刻度字体大小
    plt.yticks(fontsize=12)  # 设置 y 轴刻度字体大小
    plt.legend(fontsize=9)  # 显示图例

    # 移除辅助虚线
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.tight_layout()  # 调整布局以适应图表内容
    plt.savefig(os.path.join(result_path, filename))
    plt.close()  # 关闭图表以释放资源

def plot_lines_with_breaks(x,lists,filename,x_label, y_label,result_path):
    plt.figure(figsize=(7, 5))
    # 设置图表样式和字体
    plt.style.use('default')
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.facecolor'] = 'white'  # 设置图表背景为白色
    plt.rcParams['axes.edgecolor'] = 'black'  # 设置坐标轴颜色为黑色
    plt.rcParams['axes.grid'] = False  # 禁用网格
    # 初始化全局颜色变量
    # 创建一个颜色列表，其长度与lists中的列表数量相同
    colors = plt.get_cmap('viridis')(np.linspace(0, 1, len(lists)))
    colors = ['red', 'green', 'orange', 'blue', 'black', 'purple']
    list_name = ["[18]'s method","Simulate Gyroscope","[50]'s method","Proposed method"]
    for i, y_list in enumerate(lists, start=1):
        meet_None=False
        line_to_plot = []
        points_to_plot =[]
        for j, y in enumerate(y_list):
            if y[2] is None:
                if meet_None is False:
                    if line_to_plot:
                        plt.plot(x[:j], [point[2] for point in line_to_plot], label=list_name[i-1], color=colors[i-1],
                                 marker='o', linestyle='-',linewidth=1.5, markersize=2)
                    meet_None = True
            else:
                if meet_None is True:
                    points_to_plot.append((x[j],y[2]))
                else:
                    line_to_plot.append(y)
                    if j == len(y_list)-1:
                        plt.plot(x[:], [point[2] for point in line_to_plot], label=list_name[i-1], color=colors[i-1],
                                 marker='o', linestyle='-',linewidth=1.5, markersize=2)
        # 使用 zip 函数来分离 x 和 y 坐标
        if points_to_plot:
            x_coords, y_coords = zip(*points_to_plot)
            #plt.plot(x_coords, y_coords, 'o', label=f'Line {i} Break Point')
            plt.plot(x_coords, y_coords, 'o', label='',color=colors[i-1])

    plt.xlabel(x_label,fontsize=10)
    plt.ylabel(y_label,fontsize=10)
    #plt.title('Line Chart with Breaks and Markers')
    plt.xticks(fontsize=10)  # 设置 x 轴刻度字体大小
    plt.yticks(fontsize=10)  # 设置 y 轴刻度字体大小
    plt.legend(fontsize=8)
    #plt.grid(True)
    # 应用对数尺度
    #plt.xscale('log')  # 对数尺度应用于 x 轴
    #plt.yscale('log')  # 对数尺度应用于 y 轴
    # 移除辅助虚线
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()  # 调整布局以适应图表内容
    plt.savefig(os.path.join(result_path, filename))
    plt.show()
    plt.close()  # 关闭图表以释放资源

def plot_tuples_adjust(list1, list2, list3, list4,x_label,y_label,path,filename,log=True):
    """绘制四个列表中元组的线图。

    每个列表中的元组的第一个元素作为横坐标，第二个元素作为纵坐标。
    """
    plt.figure(figsize=(5, 3.5))
    # 设置图表样式和字体
    plt.style.use('default')
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.facecolor'] = 'white'  # 设置图表背景为白色
    plt.rcParams['axes.edgecolor'] = 'black'  # 设置坐标轴颜色为黑色
    plt.rcParams['axes.grid'] = False  # 禁用网格
    colors = ['red', 'green', 'orange', 'blue', 'black', 'purple']
    list_name = ["[18]'s method", "RL", "[50]'s method", "Proposed method"]

    # 分别提取每个列表中的元组的横纵坐标
    x1, y1 = zip(*list1)
    x2, y2 = zip(*list2)
    x3, y3 = zip(*list3)
    x4, y4 = zip(*list4)

    # 绘制四条线
    plt.plot(x1, y1, label=list_name[0], color=colors[0],marker='o', linestyle='-',linewidth=1.5, markersize=2)
    plt.plot(x2, y2, label=list_name[1], color=colors[1],marker='o', linestyle='-',linewidth=1.5, markersize=2)
    plt.plot(x3, y3, label=list_name[2], color=colors[2],marker='o', linestyle='-',linewidth=1.5, markersize=2)
    plt.plot(x4, y4, label=list_name[3], color=colors[3],marker='o', linestyle='-',linewidth=1.5, markersize=2)
    plt.xlabel(x_label,fontsize=12)
    plt.ylabel(y_label,fontsize=12)
    plt.xticks(fontsize=12)  # 设置 x 轴刻度字体大小
    plt.yticks(fontsize=12)  # 设置 y 轴刻度字体大小
    plt.legend(fontsize=9)
    if log:
        plt.yscale('log')  # 对数尺度应用于 y 轴
    # 移除辅助虚线
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()  # 调整布局以适应图表内容
    plt.savefig(os.path.join(path, filename))
    # 显示图表
    plt.show()
    plt.close()  # 关闭图表以释放资源


def load_csv_to_tuples(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过表头
        result = []
        for row in reader:
            # 尝试将每个元素转换为浮点数，如果失败则保留原始字符串
            converted_row = []
            for item in row:
                try:
                    converted_row.append(float(item))
                except ValueError:
                    converted_row.append(item)
            result.append(tuple(converted_row))
        return result

def main1():
    APART_RESULT_PATH = r'D:\programing\research_code\centernet-hybrid-withoutmv-predict\test\result\apart_result'

    # 载入CSV文件并恢复成列表
    f1_scores_list = load_csv_to_tuples(os.path.join(APART_RESULT_PATH, 'f1_scores.csv'))
    recall_scores_list = load_csv_to_tuples(os.path.join(APART_RESULT_PATH, 'recall_scores.csv'))
    precision_scores_list = load_csv_to_tuples(os.path.join(APART_RESULT_PATH, 'precision_scores.csv'))

    print(f1_scores_list)

    plot_and_save_objects(recall_scores_list, precision_scores_list, f1_scores_list, 'w (°/s)',
                          'Objective detection metrics (%)',
                          'Objective detection adjust.png', APART_RESULT_PATH, 2)

def main2():
    amount = 20
    start = 0
    end = 7
    path = r'D:\programing\data\replicate\angular_w_40_0_8'
    path = f'D:/programing/data/replicate/angular_w_{amount}_{start}_{end}'
    x=generate_uniform_distribution(start, end, amount)
    print(x)
    lr_ac = load_csv_to_tuples(os.path.join(path, 'rl_accelerated_position.csv'))
    lr_basic = load_csv_to_tuples(os.path.join(path, 'rl_basic_position.csv'))
    wiener = load_csv_to_tuples(os.path.join(path, 'wiener_position.csv'))
    mine = load_csv_to_tuples(os.path.join(path, 'mine_position.csv'))
    lists = [lr_ac,lr_basic,wiener,mine]
    plot_lines_with_breaks(x, lists, 'bias', 'w (°/s)', 'deviation (pixel)', path)

def main3():
    pic_type = 'angular'
    factor = 'magnitude'
    group = 7
    start = 3
    end = 6
    amount = 200
    test_name = f'{pic_type}_{factor}_{group}_{start}_{end}_{amount}'
    replicate_path = os.path.join(r'D:\programing\data\replicate', test_name)

    lr_ac = load_csv_to_tuples(os.path.join(replicate_path, 'rl_accelerated_bias.csv'))
    lr_basic = load_csv_to_tuples(os.path.join(replicate_path, 'rl_basic_bias.csv'))
    wiener = load_csv_to_tuples(os.path.join(replicate_path, 'wiener_bias.csv'))
    mine = load_csv_to_tuples(os.path.join(replicate_path, 'mine_bias.csv'))

    lr_ac_N = load_csv_to_tuples(os.path.join(replicate_path, 'rl_accelerated_None.csv'))
    lr_basic_N = load_csv_to_tuples(os.path.join(replicate_path, 'rl_basic_None.csv'))
    wiener_N = load_csv_to_tuples(os.path.join(replicate_path, 'wiener_None.csv'))
    mine_N = load_csv_to_tuples(os.path.join(replicate_path, 'mine_None.csv'))

    plot_tuples_adjust(lr_ac,lr_basic,wiener,mine,'magnitude','deviation (pixel)',replicate_path,'bias')
    plot_tuples_adjust(lr_ac_N, lr_basic_N, wiener_N, mine_N, 'magnitude', 'None_ratio', replicate_path, 'None',False)


if __name__ == "__main__":
    main3()