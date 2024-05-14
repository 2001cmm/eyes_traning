import os
import pandas as pd


def read_csv_mean_values(directory, mean):
    """
    从指定目录中读取所有CSV文件，并返回它们的均值数据列表。

    参数：
    directory: str
        包含CSV文件的目录路径。

    返回：
    list
        包含所有CSV文件均值数据的列表。
    """
    # 存储均值数据的列表
    mean_values = []

    # 获取目录下所有文件列表
    files = os.listdir(directory)

    # 按文件名排序
    files.sort()

    # 遍历文件列表
    for file_name in files:
        if file_name.endswith('.csv'):
            # 构建CSV文件的完整路径
            file_path = os.path.join(directory, file_name)

            # 使用pandas读取CSV文件数据
            data = pd.read_csv(file_path)

            # 提取数据的均值列，并将数据添加到列表中
            data_mean = data[mean]
            mean_values.extend(data_mean)

    return mean_values


# 调用函数并传入目录路径
# directory = r'D:\桌面\毕设\数据处理需要\Annotation-task1'
# mean_values_list = read_csv_mean_values(directory)
