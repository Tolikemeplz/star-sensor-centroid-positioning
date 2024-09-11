import matplotlib.pyplot as plt
import numpy as np

def plot_lines_with_breaks_corrected(x, lists):
    plt.figure(figsize=(10, 6))

    for i, y_list in enumerate(lists, start=1):
        # 初始化一个列表来存储要绘制的点
        points_to_plot = []
        for j, y in enumerate(y_list):
            if y[2] is None:
                # 遇到 None，绘制之前的点，并清空点列表
                if points_to_plot:
                    plt.plot(x[:j], [point[2] for point in points_to_plot], label=f'Line {i}' if i == 1 else "")
                    points_to_plot = []
            else:
                # 如果不是 None，添加点到列表中
                points_to_plot.append(y)
                # 如果这是 None 后的第一个非 None 点，标记这个点
                if len(points_to_plot) == 1 and (j > 0 and lists[i-1][j-1][2] is None):
                    plt.plot(x[j], y[2], 'o', label=f'Line {i} Break Point')

        # 绘制最后一个线段（如果存在）
        if points_to_plot:
            plt.plot(x[:len(points_to_plot)], [point[2] for point in points_to_plot], label=f'Line {i}' if i == 1 else "")

def plot_break(x,lists):
    plt.figure(figsize=(10, 6))
    # 初始化全局颜色变量
    # 创建一个颜色列表，其长度与lists中的列表数量相同
    colors = plt.get_cmap('viridis')(np.linspace(0, 1, len(lists)))
    colors = ['red', 'green', 'orange', 'blue', 'black', 'purple']
    for i, y_list in enumerate(lists, start=1):
        meet_None=False
        line_to_plot = []
        points_to_plot =[]
        for j, y in enumerate(y_list):
            if y[2] is None:
                if meet_None is False:
                    if line_to_plot:
                        plt.plot(x[:j], [point[2] for point in line_to_plot], label=f'Line {i}', color=colors[i-1])
                    meet_None = True
            else:
                if meet_None is True:
                    points_to_plot.append((x[j],y[2]))
                else:
                    line_to_plot.append(y)
                    if j == len(y_list)-1:
                        plt.plot(x[:], [point[2] for point in line_to_plot], label=f'Line {i}', color=colors[i-1])
        # 使用 zip 函数来分离 x 和 y 坐标
        if points_to_plot:
            x_coords, y_coords = zip(*points_to_plot)
            #plt.plot(x_coords, y_coords, 'o', label=f'Line {i} Break Point')
            plt.plot(x_coords, y_coords, 'o', label='',color=colors[i-1])

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Line Chart with Breaks and Markers')
    plt.legend()
    plt.grid(True)
    plt.show()

# 示例数据
n = 5
x = np.arange(n)
lists_with_none = [
    [(1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12), (13, 14, 15)],
    [(2, 3, 4), (5, 6, 7), (8, 9, 10), (11, 12, 13), (14, 15, 16)],
    [(3, 4, 5), (6, 7, 8), (9, 10, 11), (12, 13, 14), (15, 16, 17)],
    [(4, 5, 6), (7, 8, 9), (10, 11, 12), (13, 14, 15), (16, 17, 18)]
]

# 使用修正后的函数重新绘制图表
#plot_lines_with_breaks_corrected(x, lists_with_none)
plot_break(x,lists_with_none)
