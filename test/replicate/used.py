import numpy as np
import cv2
import math
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy import ndimage



def find_slope_point(segment, slope, direction):
    """
    Find the index of the first point where the slope is greater than the given value
    for the 'left' direction or less than the negative of the given value for the 'right' direction.

    Parameters:
    - segment: 1D array, the data segment to search within
    - slope: the slope value to compare against
    - direction: 'left' or 'right', direction to search for the slope

    Returns:
    - Index of the first point with the appropriate slope or None if not found
    """
    for i in range(1, len(segment)):
        current_slope = (segment[i] - segment[i - 1]) / (i - (i - 1))

        if direction == 'left' and current_slope > slope:
            # 左边这个还有点问题，应该是从右到左看那个小于slope
            return i - 1
        elif direction == 'right' and current_slope > -slope:
            return i - 1
    return None

def find_first_extrema_corrected_again(column, peak_index, direction='left', slope_a=0.01):
    """
    Find the index of the first extremum (minimum or maximum) to the left or right of a given peak.
    If no extremum is found, find the first point with a slope of 'a' or '-a' depending on the direction.

    Parameters:
    - column: 1D array, the data in which to find the extrema
    - peak_index: index of the peak around which to search
    - direction: 'left' or 'right', direction to search for the extrema
    - slope_a: the slope value to find if no extremum is found

    Returns:
    - Index of the first extremum found or the first point with the given slope
    """
    if direction == 'left':
        segment = column[:peak_index]
        extrema = find_peaks(-segment)[0]
        if len(extrema) > 0:
            return extrema[-1]  # Return the closest extrema to the peak
        elif slope_a is not None:
            slope_point_index = find_slope_point(segment, slope_a, 'left')
            print('峰值左边为: ',slope_point_index)
            return find_slope_point(segment, slope_a, 'left')
    elif direction == 'right':
        segment = column[peak_index:]
        extrema = find_peaks(-segment)[0]
        extrema = [x + peak_index for x in extrema]  # Adjust the index for the segment
        if len(extrema) > 0:
            return extrema[0]  # Return the closest extrema to the peak
        elif slope_a is not None:
            slope_point_index = find_slope_point(segment, slope_a, 'right')
            print('峰值右边为: ', slope_point_index)
            return slope_point_index + peak_index + 1 if slope_point_index is not None else None
    else:
        raise ValueError("Direction must be 'left' or 'right'")

    return None  # No extrema or slope point found

def find_peak_width_corrected(projections, a):
    """
    Find the width of the central peak in the specified column of projections,
    corrected to find the first extrema on both sides.

    Parameters:
    - projections: 2D array, result of Radon transform
    - a: index of the column to analyze

    Returns:
    - Width of the central peak
    """
    # Extract the specified column
    column = projections[:, a]

    # Find peaks in the column
    peaks, _ = find_peaks(column)

    # Find the central peak (assuming it's the highest)
    central_peak_index = np.argmax(column[peaks])

    # Find the central peak's index
    central_peak = peaks[central_peak_index]

    # Find the first minimum to the left and right of the central peak
    left_min_index = find_first_extrema_corrected_again(column, central_peak, direction='left')
    right_min_index = find_first_extrema_corrected_again(column, central_peak, direction='right')

    # Calculate the width as the difference between the right and left extrema
    width = right_min_index - left_min_index if left_min_index is not None and right_min_index is not None else None
    return width

def z_function(image, a=5, c=12):
    """Z-function for nonlinear gray stretching."""
    image = image.astype(float)
    # 归一化图像
    image_max = np.max(image)
    if image_max > 0:
        image /= image_max
        image = image*254
    image[image <= a] = 0
    image[(a < image) & (image <= (a + c)/2)] = 2*(image[(a < image) & (image <= (a + c)/2)] - c) / (c - a) ** 2
    mask = ((a + c) / 2 < image) & (image <= c)
    image[mask] = 1 - 2 * (image[mask] - a) ** 2 / ((c - a) ** 2)
    image[image > c] = 1
    return image




def double_threshold_mask(image, global_thresh=0.08, local_thresh=0.12):
    """Double threshold mask with visualization of global_mask and final local_mask."""
    max_val = np.max(image)
    global_mask = image > max_val * global_thresh
    local_masks = []

    labels, num_features = ndimage.label(global_mask)
    for i in range(1, num_features + 1):
        mask = (labels == i)
        if np.sum(mask) > 0:
            local_max = np.max(image[mask])
            local_mask = np.zeros_like(image, dtype=bool)
            local_mask[mask] = image[mask] > local_max * local_thresh
            local_masks.append(local_mask)

    # Initialize the final local_mask
    local_mask = np.zeros_like(image, dtype=float)
    for m in local_masks:
        local_mask += m.astype(float)

    # Visualizing the global_mask and final local_mask side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(global_mask, cmap='gray')
    axes[0].set_title('Global Mask')
    axes[0].axis('off')

    axes[1].imshow(local_mask, cmap='gray')
    axes[1].set_title('Final Local Mask')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

    return global_mask * local_mask

def find_blur_angle(projections, theta):
    max_index = np.unravel_index(np.argmax(projections),projections.shape)
    # print('在find_blur_angle中max_index为', max_index)
    # print('max_idex的形状为', max_index.shape)
    blur_angle = theta[max_index[1]]  # Use the first max index along columns
    return blur_angle, max_index[1]

def motion_blur_kernel(kernel_size, angle):
    kernel_size = round(kernel_size)
    kernel = np.zeros((kernel_size, kernel_size))
    center = (kernel_size - 1) // 2
    kernel[center, :] = np.ones(kernel_size)
    rotation_matrix = cv2.getRotationMatrix2D((center, center), angle, 1)
    kernel = cv2.warpAffine(kernel, rotation_matrix, (kernel_size, kernel_size))
    kernel = kernel / kernel_size  # 归一化
    # If the original kernel_size was even, add a row and column of zeros
    kernel_origin = kernel
    if (kernel_size) % 2 == 0:
        kernel = np.pad(kernel, ((0, 1), (0, 1)), mode='constant')

    return kernel,kernel_origin

def motion_blur_kernel_fullsize(image_size, blur_length, angle):
    blur_length = round(blur_length)
    M, N = image_size
    kernel = np.zeros(image_size)
    center = (M - 1) // 2

    # 创建一个水平线段的PSF
    # kernel[center - blur_length // 2:center + blur_length // 2 + 1, center] = 1
    kernel[center, center - blur_length // 2:center + blur_length // 2 + 1] = 1

    # 获取旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D((center, center), angle, 1)

    # 旋转PSF以创建运动模糊效果
    kernel = cv2.warpAffine(kernel, rotation_matrix, (M, N))

    # 归一化PSF
    kernel = kernel / np.sum(kernel)

    return kernel

def calculate_line_angle(x1, y1, x2, y2):
    # 计算斜率
    m = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')

    # 计算弧度
    theta_radians = math.atan(m)

    # 转换为度
    theta_degrees = math.degrees(theta_radians)

    # # 确定角度方向
    # if x2 < x1:
    #     theta_degrees += 180
    #
    # # 将角度标准化到 0 到 180 度之间
    # if theta_degrees < 0:
    #     theta_degrees += 180
    # elif theta_degrees > 180:
    #     theta_degrees -= 180

    return abs(theta_degrees)



def w_1(h,height):
    hw = h[1:,:-1]
    # print('hw:',hw,"\n")
    rows, cols = hw.shape
    result = np.zeros_like(hw)

    for i in range(rows):
        for j in range(cols):
            result[i, j] = hw[i:, :j + 1].sum()
    return result

def w_8(h,height):
    hw = h[:,:-1]
    # print('hw:', hw,"\n")
    h_rows,h_cols = h.shape
    result_shape = (height-h_rows*2+2,h_cols-1)
    result = np.zeros(result_shape)
    rows, cols = result_shape

    for i in range(rows):
        for j in range(cols):
            result[i,j] = hw[:,:j+1].sum()
    return result

def w_7(h,height):
    hw = h[:-1,:-1]
    # print('hw:', hw, "\n")
    rows, cols = hw.shape
    result = np.zeros_like(hw)

    for i in range(rows):
        for j in range(cols):
            result[i, j] = hw[:i+1, :j + 1].sum()

    return result

def w_2(h,height):
    hw = h[1:,:]
    # print('hw:', hw, "\n")
    h_rows, h_cols = h.shape
    result_shape = (h_rows-1,height-h_cols*2+2)
    result = np.zeros(result_shape)
    rows, cols = result_shape

    for i in range(rows):
        for j in range(cols):
            result[i,j] = hw[i:,:].sum()

    return result

def w_6(h,height):
    hw = h[:-1,:]
    # print('hw:', hw, "\n")
    h_rows, h_cols = h.shape
    result_shape = (h_rows - 1, height - h_cols * 2 + 2)
    result = np.zeros(result_shape)
    rows, cols = result_shape

    for i in range(rows):
        for j in range(cols):
            result[i,j] = hw[:i+1, :].sum()

    return result

def w_3(h,height):
    hw = h[1:,1:]
    # print('hw:', hw, "\n")
    rows, cols = hw.shape
    result = np.zeros_like(hw)

    for i in range(rows):
        for j in range(cols):
            result[i,j] = hw[i:,j:].sum()
    return result

def w_4(h,height):
    hw = h[:,1:]
    # print('hw:', hw, "\n")
    h_rows, h_cols = h.shape
    result_shape = (height - h_rows * 2 + 2, h_cols - 1)
    result = np.zeros(result_shape)
    rows, cols = result_shape

    for i in range(rows):
        for j in range(cols):
            result[i,j] = hw[:,j:].sum()
    return result

def w_5(h,height):
    hw = h[:-1,1:]
    # print('hw:', hw, "\n")
    rows, cols = hw.shape
    result = np.zeros_like(hw)

    for i in range(rows):
        for j in range(cols):
            result[i,j] = hw[:i+1,j:].sum()
    return result

def optimal_window (g,h):
    height,_ = g.shape
    d,_ = h.shape
    window = np.zeros(g.shape)
    window[-(d-1):,:d-1] = w_1(h,height)
    window[-(d-1):,d-1:-(d-1)] = w_2(h,height)
    window[-(d-1):,-(d-1):] = w_3(h,height)
    window[d-1:-(d-1),-(d-1):] = w_4(h,height)
    window[:d-1,-(d-1):] = w_5(h,height)
    window[:d-1,d-1:-(d-1)] = w_6(h,height)
    window[:d-1,:d-1] = w_7(h,height)
    window[d-1:-(d-1),:d-1] = w_8(h,height)
    window[d-1:-(d-1),d-1:-(d-1)] = 1
    return window

def motion_blur_kernel_win(kernel_size, angle):
    kernel = np.zeros((kernel_size, kernel_size))
    center = (kernel_size - 1) // 2
    kernel[center, :] = np.ones(kernel_size)
    rotation_matrix = cv2.getRotationMatrix2D((center, center), angle, 1)
    kernel = cv2.warpAffine(kernel, rotation_matrix, (kernel_size, kernel_size))
    kernel = kernel / kernel_size  # 归一化
    return kernel


def process_image(binary_img, original_img):
    # 将binary_img的布尔值转换为uint8类型，True变为255，False变为0
    binary_img = np.uint8(binary_img) * 255

    # 找到连通域
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 如果没有连通域，返回None
    if len(contours) == 0:
        print("No contours found.")
        return None

    # 如果只有一个连通域
    if len(contours) == 1:
        contour = contours[0]
        # 判断像素个数
        if cv2.contourArea(contour) < 16:
            # 初始化膨胀操作的核
            kernel = np.ones((3, 3), np.uint8)
            # 创建一个与原图同样大小的空白图像
            dilated_contour_img = np.zeros_like(binary_img)
            # 当选中的连通域像素个数小于16时，进行膨胀操作
            while cv2.contourArea(contour) < 16:
                cv2.drawContours(dilated_contour_img, [contour], -1, 255, thickness=cv2.FILLED)
                dilated_contour_img = cv2.dilate(dilated_contour_img, kernel, iterations=1)
                contours, _ = cv2.findContours(dilated_contour_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contour = contours[0]
        # 创建一个全黑的图像，并将膨胀后的连通域画上去
        result_img = np.zeros_like(binary_img)
        cv2.drawContours(result_img, [contour], -1, 255, thickness=cv2.FILLED)
        return result_img

    # 如果有多个连通域
    else:
        # 去掉像素个数大于144个的连通域
        filtered_contours = [c for c in contours if cv2.contourArea(c) <= 144]

        # 计算剩余连通域的灰度值之和
        max_sum = 0
        selected_contour = None
        for contour in filtered_contours:
            mask = np.zeros_like(binary_img)
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
            gray_sum = np.sum(original_img[mask == 255])
            if gray_sum > max_sum:
                max_sum = gray_sum
                selected_contour = contour

        # 如果找到的连通域像素个数小于16，则进行膨胀操作
        if cv2.contourArea(selected_contour) < 16:
            kernel = np.ones((3, 3), np.uint8)
            dilated_contour_img = np.zeros_like(binary_img)
            while cv2.contourArea(selected_contour) < 16:
                cv2.drawContours(dilated_contour_img, [selected_contour], -1, 255, thickness=cv2.FILLED)
                dilated_contour_img = cv2.dilate(dilated_contour_img, kernel, iterations=1)
                contours, _ = cv2.findContours(dilated_contour_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                selected_contour = contours[0]

        # 创建一个全黑的图像，并将膨胀后的连通域画上去
        result_img = np.zeros_like(binary_img)
        cv2.drawContours(result_img, [selected_contour], -1, 255, thickness=cv2.FILLED)
        return result_img

def calculate_weighted_centroid(original_img, thresholded_img):
    """
    使用原图的灰度值加权计算连通域的质心坐标。

    :param original_img: 原始灰度图像
    :param thresholded_img: 阈值分割后的二值图像
    :return: 加权质心坐标
    """
    # 确保阈值分割图像是二值的

    # 如果thresholded_img是None，则返回None
    if thresholded_img is None:
        return None, None

    _, binary_img = cv2.threshold(thresholded_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 寻找连通域
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img, connectivity=8)

    # 跳过背景，找到最大的连通域（基于面积）
    max_area = 0
    max_area_idx = 0
    for i in range(1, num_labels):  # 从1开始，因为0是背景
        if stats[i, cv2.CC_STAT_AREA] > max_area:
            max_area = stats[i, cv2.CC_STAT_AREA]
            max_area_idx = i

    # 创建一个与原图同样大小的零矩阵，用于存储连通域的灰度值
    component_pixels = np.zeros_like(original_img, dtype=np.float64)

    # 将连通域的灰度值复制到component_pixels中
    component_pixels[labels == max_area_idx] = original_img[labels == max_area_idx]

    # 计算加权质心坐标
    total_weight = np.sum(component_pixels)
    weighted_x = np.sum(component_pixels * (np.arange(component_pixels.shape[0]).reshape(-1, 1))) / total_weight
    weighted_y = np.sum(component_pixels * (np.arange(component_pixels.shape[1]))) / total_weight

    return weighted_x, weighted_y

def generate_uniform_distribution(start, end, count):
    return [start + (end - start) * i / (count - 1) for i in range(count)]

def plot_lines_with_breaks(x,lists):
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
    # 应用对数尺度
    #plt.xscale('log')  # 对数尺度应用于 x 轴
    plt.yscale('log')  # 对数尺度应用于 y 轴
    plt.show()


def plot_tuples(list1, list2, list3, list4):
    """绘制四个列表中元组的线图。

    每个列表中的元组的第一个元素作为横坐标，第二个元素作为纵坐标。
    """
    # 分别提取每个列表中的元组的横纵坐标
    x1, y1 = zip(*list1)
    x2, y2 = zip(*list2)
    x3, y3 = zip(*list3)
    x4, y4 = zip(*list4)

    # 绘制四条线
    plt.plot(x1, y1, label='List 1')
    plt.plot(x2, y2, label='List 2')
    plt.plot(x3, y3, label='List 3')
    plt.plot(x4, y4, label='List 4')

    # 添加图例
    plt.legend()
    plt.yscale('log')  # 对数尺度应用于 y 轴
    # 显示图表
    plt.show()




def replace_values(lst,s):
    modified_list = []
    for sublist in lst:
        modified_sublist = []
        for elem in sublist:
            if isinstance(elem, tuple):
                modified_tuple = tuple(None if x > s else x for x in elem)
                modified_sublist.append(modified_tuple)
            else:
                modified_sublist.append(None if elem > s else elem)
        modified_list.append(modified_sublist)
    return modified_list

def average_of_third_elements(tuple_list,amount):
    """计算列表中每个元组第三个元素的平均值，跳过None值，并返回None的个数"""
    sum_of_third_elements = 0
    count_of_third_elements = 0
    none_count = 0

    for t in tuple_list:
        if t[2] is None:
            none_count += 1
        else:
            sum_of_third_elements += t[2]
            count_of_third_elements += 1

    # 如果所有元素都是None，则避免除以0
    if count_of_third_elements == 0:
        average = None
    else:
        average = sum_of_third_elements / count_of_third_elements

    return average, none_count/amount

