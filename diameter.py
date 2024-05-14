import os
import pandas as pd
import numpy as np
import jisuan  # 假设这是之前提到的自定义模块
from buchongshuju import expand_features


def process_pupil_data(folder_path, calc_stat_func):
    """
    处理瞳孔数据文件夹中的数据。

    :param folder_path: 数据文件夹的路径
    :param calc_stat_func: 计算统计量的函数
    :return: 处理结果的字典
    """
    # 初始化存储结果的字典
    results = {}

    # 遍历文件夹中的所有文件
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            # 构建完整的文件路径
            file_path = os.path.join(folder_path, file_name)
            # 读取CSV文件
            data = pd.read_csv(file_path)
            # 按eye_id和method分组
            grouped_data = data.groupby(['eye_id', 'method'])

            # 遍历每个分组
            for (eye_id, method), group in grouped_data:
                # 只对eye_id为0或1的数据进行处理
                if eye_id in [0, 1]:
                    x = group['diameter']
                    time = group['pupil_timestamp']
                    # 计算差分
                    x_diff = np.diff(x)
                    time_diff = np.diff(time)
                    # 计算瞳孔直径变化率
                    diameter_change = x_diff / time_diff
                    # 构建存储键
                    key = (eye_id, method)
                    # 如果该组合的数据尚未存储，则初始化存储结构
                    if key not in results:
                        results[key] = {
                            'diameter_changes': [],
                            'duration_lens': [],
                            'diameter_calculate': [],
                            'diameter_changes_calculate': [],
                            'diameter_changes_calculate_kuozhan': []
                        }
                    # 将结果存储到字典中
                    results[key]['diameter_changes'].extend(diameter_change)
                    # 计算统计量并存储
                    results[key]['diameter_changes_calculate'].append(calc_stat_func(diameter_change))
                    results[key]['diameter_calculate'].append(calc_stat_func(x))

    return results


# # 使用示例
# folder_path1 = r'D:\桌面\毕设\数据处理需要\pupill-task1'
# processed_results = process_pupil_data(folder_path1, jisuan.calc_stat)
#
# # 打印特定eye_id和method组合的计算结果
# if (0, '2d c++') in processed_results:
#     print(processed_results[0, '2d c++']['diameter_changes_calculate'])
