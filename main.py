import csv
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

import kmeans
import ra_tree
import detect
import table_crate
from buchongshuju import expand_features
from normalize import normalize_features
from svm import train_and_predict_svm
from svm_classfiver import train_svm_classifier
import jisuan
from diameter import process_pupil_data
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def normalize_columns(matrix):
    min_vals = np.min(matrix, axis=0)
    max_vals = np.max(matrix, axis=0)
    normalized_matrix = (matrix - min_vals) / (max_vals - min_vals)
    return normalized_matrix


"""
输入到网络里矩阵的样子
     |持续时间的期望，持续时间的标准差，持续时间的偏度，持续时间的峰度。。。。|
 x = |                                                          |
     |
     |
"""
###任务1的特征提取
# path = "123.csv"
# data = pd.read_csv(path)
# print(data)
# 获取CSV文件所在的文件夹路径
# folder_path1 = sorted(os.listdir('D:\桌面\毕设\数据处理需要\gaze-task1'))
folder_path1 = 'D:\桌面\毕设\数据处理需要\gaze-task1'

duration_len1 = []

duration_all1_task1_calculate = []
speed1_task1_calculate = []
speed_len1 = []

distance1_task4_calculate = []
files1 = os.listdir(folder_path1)
files1.sort()

print('**********************开始处理训练集*******************************')
for file_name in files1:
    if file_name.endswith('.csv'):
        # 构建CSV文件的完整路径
        file_path1 = os.path.join(folder_path1, file_name)

        # print(file_path1)
        # 使用pandas读取CSV文件数据
        data = pd.read_csv(file_path1)
        duration_max1 = []
        speed1 = []
        duration_all1 = []
        distance1 = []
        data_len = len(data)
        # print(data_len)
        x = data['norm_pos_x']
        y = data['norm_pos_y']
        time = data['gaze_timestamp']
        x, y, time = detect.remove_missing(x, y, time, missing=-1)
        x.index = list(range(len(x)))
        y.index = list(range(len(y)))
        time.index = list(range(len(time)))
        # 持续时间提取
        Sfix, Efix, x1 = detect.fixation_detection(x, y, time, missing=0, maxdist=0.15, mindur=1)
        Efix = np.array(Efix)
        duration_max1.append(Efix[:, 2])
        duration_all1 = np.concatenate(duration_max1)
        # 注视点个数提取
        duration_len1.append(len(Efix))
        # duration_len1 = np.concatenate(duration_len1)
        # 扫视速度提取
        Ssac, Esac = detect.saccade_detection(x, y, time, missing=0.0, minlen=2, maxvel=6.4, maxacc=40)
        Easc = np.array(Esac)
        # print(Easc)
        speed1.extend((((Easc[:, 3] - Easc[:, 5]) ** 2 + (Easc[:, 4] - Easc[:, 6]) ** 2) ** 0.5 / Easc[:, 2]) * 1000)
        duration_all1_task1_calculate.append(jisuan.calc_stat(duration_all1))
        # speed1 = np.concatenate(speed1)
        speed1_task1_calculate.append(jisuan.calc_stat(speed1))
        # 扫视个数
        speed_len1.append(len(Esac))
        # 扫视距离（欧式距离）
        distance1.extend(((Easc[:, 3] - Easc[:, 5]) ** 2 + (Easc[:, 4] - Easc[:, 6]) ** 2) ** 0.5)
        distance1_task4_calculate.append(jisuan.calc_stat(distance1))

# speed1 = np.concatenate(speed1)
duration_len1 = np.array(duration_len1)
duration_len1 = duration_len1[:, np.newaxis]
speed_len1 = np.array(speed_len1)
speed_len1 = speed_len1[:, np.newaxis]
merged_feature_matrix1 = np.hstack((duration_all1_task1_calculate, speed1_task1_calculate,
                                    distance1_task4_calculate, duration_len1, speed_len1
                                    ))

print('task1处理成功')

# sys.exit()
###任务2的特征提取
# folder_path2 = sorted(os.listdir('D:\桌面\毕设\数据处理需要\gaze-task2'))
folder_path2 = 'D:\桌面\毕设\数据处理需要\gaze-task2'

duration_len2 = []

duration_all2_task2_calculate = []
speed2_task2_calculate = []
speed_len2 = []

distance2_task4_calculate = []
files2 = os.listdir(folder_path2)
files2.sort()
duration_all2_task2_calculate_kuozhan = []
distance2_task4_calculate_kuozhan = []
speed2_task2_calculate_kuozhan = []
for file_name in files2:
    if file_name.endswith('.csv'):
        # 构建CSV文件的完整路径
        file_path2 = os.path.join(folder_path2, file_name)
        # print(file_path)
        # 使用pandas读取CSV文件数据
        data = pd.read_csv(file_path2)
        duration_max2 = []
        speed2 = []
        duration_all2 = []
        distance2 = []
        data_len = len(data)
        # print(data_len)
        x = data['norm_pos_x']
        y = data['norm_pos_y']
        time = data['gaze_timestamp']
        x, y, time = detect.remove_missing(x, y, time, missing=-1)
        x.index = list(range(len(x)))
        y.index = list(range(len(y)))
        time.index = list(range(len(time)))
        # 持续时间提取
        Sfix, Efix, x1 = detect.fixation_detection(x, y, time, missing=0, maxdist=0.15, mindur=1.3)
        Efix = np.array(Efix)
        duration_max2.append(Efix[:, 2])
        duration_all2 = np.concatenate(duration_max2)
        # 注视点个数
        duration_len2.append(len(Efix))
        # duration_len2 = np.concatenate(duration_len2)
        # 扫视速度提取
        Ssac, Esac = detect.saccade_detection(x, y, time, missing=0.0, minlen=2, maxvel=6.5, maxacc=40)
        Easc = np.array(Esac)
        speed2.extend((((Easc[:, 3] - Easc[:, 5]) ** 2 + (Easc[:, 4] - Easc[:, 6]) ** 2) ** 0.5 / Easc[:, 2]) * 1000)
        speed2_task2_calculate.append(jisuan.calc_stat(speed2))
        duration_all2_task2_calculate.append(jisuan.calc_stat(duration_all2))
        # 扫视个数
        speed_len2.append(len(Esac))
        # 扫视距离（欧式距离）
        distance2.extend(((Easc[:, 3] - Easc[:, 5]) ** 2 + (Easc[:, 4] - Easc[:, 6]) ** 2) ** 0.5)
        distance2_task4_calculate.append(jisuan.calc_stat(distance2))
# speed2 = np.concatenate(speed2)
duration_len2 = np.array(duration_len2)
duration_len2 = duration_len2[:, np.newaxis]
speed_len2 = np.array(speed_len2)
speed_len2 = speed_len2[:, np.newaxis]
merged_feature_matrix2 = np.hstack((duration_all2_task2_calculate, speed2_task2_calculate,
                                    distance2_task4_calculate, duration_len2, speed_len2
                                    ))

print('task2处理完成')
###任务3的特征提取
# folder_path3 = sorted(os.listdir('D:\桌面\毕设\数据处理需要\gaze-task3'))
folder_path3 = 'D:\桌面\毕设\数据处理需要\gaze-task3'

duration_len3 = []

duration_all3_task3_calculate = []
speed3_task3_calculate = []
speed_len3 = []

distance3_task4_calculate = []
files3 = os.listdir(folder_path3)
files3.sort()
duration_all3_task3_calculate_kuozhan = []
distance3_task4_calculate_kuozhan = []
speed3_task3_calculate_kuozhan = []
for file_name in files3:
    if file_name.endswith('.csv'):
        # 构建CSV文件的完整路径
        file_path3 = os.path.join(folder_path3, file_name)
        # print(file_path)
        # 使用pandas读取CSV文件数据
        data = pd.read_csv(file_path3)
        duration_max3 = []
        speed3 = []
        duration_all3 = []
        distance3 = []
        data_len = len(data)
        # print(data_len)
        x = data['norm_pos_x']
        y = data['norm_pos_y']
        time = data['gaze_timestamp']
        x, y, time = detect.remove_missing(x, y, time, missing=-1)
        x.index = list(range(len(x)))
        y.index = list(range(len(y)))
        # 持续时间提取
        Sfix, Efix, x1 = detect.fixation_detection(x, y, time, missing=0, maxdist=0.15, mindur=2)
        Efix = np.array(Efix)
        duration_max3.append(Efix[:, 2])
        duration_all3 = np.concatenate(duration_max3)
        # 注视点个数
        duration_len3.append(len(Efix))
        # duration_len3 = np.concatenate(duration_len3)
        # 扫视速度提取
        Ssac, Esac = detect.saccade_detection(x, y, time, missing=0.0, minlen=2, maxvel=7, maxacc=40)
        Easc = np.array(Esac)
        speed3.extend((((Easc[:, 3] - Easc[:, 5]) ** 2 + (Easc[:, 4] - Easc[:, 6]) ** 2) ** 0.5 / Easc[:, 2]) * 1000)
        duration_all3_task3_calculate.append(jisuan.calc_stat(duration_all3))
        speed3_task3_calculate.append(jisuan.calc_stat(speed3))
        # 扫视个数
        speed_len3.append(len(Esac))
        # 扫视距离（欧式距离）
        distance3.extend(((Easc[:, 3] - Easc[:, 5]) ** 2 + (Easc[:, 4] - Easc[:, 6]) ** 2) ** 0.5)
        distance3_task4_calculate.append(jisuan.calc_stat(distance3))

# speed2 = np.concatenate(speed2)
duration_len3 = np.array(duration_len3)
duration_len3 = duration_len3[:, np.newaxis]
speed_len3 = np.array(speed_len3)
speed_len3 = speed_len3[:, np.newaxis]
merged_feature_matrix3 = np.hstack((duration_all3_task3_calculate, speed3_task3_calculate,
                                    distance3_task4_calculate, duration_len3, speed_len3
                                    ))

print('task3处理完成')

###任务4的特征提取
folder_path4 = 'D:\桌面\毕设\数据处理需要\gaze-task4'
# folder_path4 = sorted(os.listdir('D:\桌面\毕设\数据处理需要\gaze-task4'))

duration_len4 = []

duration_all4_task4_calculate = []
speed4_task4_calculate = []
speed_len4 = []

distance4_task4_calculate = []
files4 = os.listdir(folder_path4)
files4.sort()
duration_all4_task4_calculate_kuozhan = []
distance4_task4_calculate_kuozhan = []
speed4_task4_calculate_kuozhan = []
for file_name in files4:
    if file_name.endswith('.csv'):
        # 构建CSV文件的完整路径
        file_path4 = os.path.join(folder_path4, file_name)
        # print(file_path)
        # 使用pandas读取CSV文件数据
        data = pd.read_csv(file_path4)
        duration_max4 = []
        speed4 = []
        duration_all4 = []
        distance4 = []
        data_len = len(data)
        # print(data_len)
        x = data['norm_pos_x']
        y = data['norm_pos_y']
        time = data['gaze_timestamp']
        x, y, time = detect.remove_missing(x, y, time, missing=-1)
        x.index = list(range(len(x)))
        y.index = list(range(len(y)))
        time.index = list(range(len(time)))
        # 注视时间提取
        Sfix, Efix, x1 = detect.fixation_detection(x, y, time, missing=0, maxdist=0.15, mindur=2)
        Efix = np.array(Efix)
        duration_max4.append(Efix[:, 2])
        duration_all4 = np.concatenate(duration_max4)
        # 注视点数量的提取
        duration_len4.append(len(Efix))
        # duration_len4 = np.concatenate(duration_len4)
        # 扫视速度提取
        Ssac, Esac = detect.saccade_detection(x, y, time, missing=0.0, minlen=2, maxvel=7, maxacc=40)
        Easc = np.array(Esac)
        speed4.extend((((Easc[:, 3] - Easc[:, 5]) ** 2 + (Easc[:, 4] - Easc[:, 6]) ** 2) ** 0.5 / Easc[:, 2]) * 1000)
        duration_all4_task4_calculate.append(jisuan.calc_stat(duration_all4))
        speed4_task4_calculate.append(jisuan.calc_stat(speed4))
        # 扫视个数
        speed_len4.append(len(Esac))
        # 扫视距离（欧式距离）
        distance4.extend(((Easc[:, 3] - Easc[:, 5]) ** 2 + (Easc[:, 4] - Easc[:, 6]) ** 2) ** 0.5)
        distance4_task4_calculate.append(jisuan.calc_stat(distance4))

# speed2 = np.concatenate(speed2)
duration_len4 = np.array(duration_len4)
duration_len4 = duration_len4[:, np.newaxis]
speed_len4 = np.array(speed_len4)
speed_len4 = speed_len4[:, np.newaxis]
merged_feature_matrix4 = np.hstack((duration_all4_task4_calculate, speed4_task4_calculate,
                                    distance4_task4_calculate, duration_len4, speed_len4
                                    ))
print('task4处理完成')
print('************瞳孔特征****************')
# 使用示例
folder_path1 = r'D:\桌面\毕设\数据处理需要\pupill-task1'
folder_path2 = r'D:\桌面\毕设\数据处理需要\pupill-task2'
folder_path3 = r'D:\桌面\毕设\数据处理需要\pupill-task3'
folder_path4 = r'D:\桌面\毕设\数据处理需要\pupill-task4'
processed_results1 = process_pupil_data(folder_path1, jisuan.calc_stat)
processed_results2 = process_pupil_data(folder_path2, jisuan.calc_stat)
processed_results3 = process_pupil_data(folder_path3, jisuan.calc_stat)
processed_results4 = process_pupil_data(folder_path4, jisuan.calc_stat)
print('**********瞳孔特征计算完成****************')

# 加入特征矩阵里
merged_feature_matrix1 = np.hstack((merged_feature_matrix1,
                                    processed_results1[0, '2d c++']['diameter_changes_calculate'],
                                    processed_results1[1, '2d c++']['diameter_changes_calculate'],
                                    processed_results1[0, '2d c++']['diameter_calculate'],
                                    processed_results1[1, '2d c++']['diameter_calculate']
                                    ))

merged_feature_matrix2 = np.hstack((merged_feature_matrix2,
                                    processed_results2[0, '2d c++']['diameter_changes_calculate'],
                                    processed_results2[1, '2d c++']['diameter_changes_calculate'],
                                    processed_results2[0, '2d c++']['diameter_calculate'],
                                    processed_results2[1, '2d c++']['diameter_calculate'],
                                    ))
merged_feature_matrix3 = np.hstack((merged_feature_matrix3,
                                    processed_results3[0, '2d c++']['diameter_changes_calculate'],
                                    processed_results3[1, '2d c++']['diameter_changes_calculate'],
                                    processed_results3[0, '2d c++']['diameter_calculate'],
                                    processed_results3[1, '2d c++']['diameter_calculate'],
                                    ))
merged_feature_matrix4 = np.hstack((merged_feature_matrix4,
                                    processed_results4[0, '2d c++']['diameter_changes_calculate'],
                                    processed_results4[1, '2d c++']['diameter_changes_calculate'],
                                    processed_results4[0, '2d c++']['diameter_calculate'],
                                    processed_results4[1, '2d c++']['diameter_calculate']
                                    ))

"""
扩展数据
"""

# merged_feature_matrix1 = expand_features(merged_feature_matrix1)
# merged_feature_matrix2 = expand_features(merged_feature_matrix2)
# merged_feature_matrix3 = expand_features(merged_feature_matrix3)
# merged_feature_matrix4 = expand_features(merged_feature_matrix4)
"""
按照心理打分表进行标签矩阵的编写
"""
#提取心理打分平均数
# tables_1 = 'D:\桌面\毕设\数据处理需要\Annotation-task1'
# tables_2 = 'D:\桌面\毕设\数据处理需要\Annotation-task2'
# tables_3 = 'D:\桌面\毕设\数据处理需要\Annotation-task3'
# tables_4 = 'D:\桌面\毕设\数据处理需要\Annotation-task4'
# mean_values_list_1 = table_crate.read_csv_mean_values(tables_1, 'mean_1')
# mean_values_list_2 = table_crate.read_csv_mean_values(tables_2, 'mean_2')
# mean_values_list_3 = table_crate.read_csv_mean_values(tables_3, 'mean_3')
# mean_values_list_4 = table_crate.read_csv_mean_values(tables_4, 'mean_4')
# mean_values = np.hstack((mean_values_list_1, mean_values_list_2, mean_values_list_3, mean_values_list_4))
# print(mean_values)
# #进行绘制折线图
# plt.plot(mean_values, 'b*--', alpha=0.5, linewidth=1, label='acc')
# plt.show()
# #使用kmeans进行一维数据的分类
# cluster_labels = kmeans.cluster_1d_data(mean_values, n_clusters=4)
# print("聚类标签：", cluster_labels)
# merged_feature_matrix = np.vstack((merged_feature_matrix1,
#                                    merged_feature_matrix2,
#                                    merged_feature_matrix3,
#                                    merged_feature_matrix4
#                                    ))
# print(merged_feature_matrix)
# print(merged_feature_matrix.shape)
# print(cluster_labels)
# print(cluster_labels.shape())
# cluster_labels = cluster_labels.reshape(-1, 1)
# merged_feature_matrix = np.hstack((merged_feature_matrix, cluster_labels))


"""""
merged_feature_matrix1
merged_feature_matrix2
merged_feature_matrix3
merged_feature_matrix4
为特征矩阵
按照任务进行编写标签矩阵
"""""
merged_feature_matrix1_len = len(merged_feature_matrix1)
task1_label = np.full((merged_feature_matrix1_len,), 1)
feature_matrix1 = np.hstack((merged_feature_matrix1, task1_label.reshape(-1, 1)))

merged_feature_matrix2_len = len(merged_feature_matrix2)
task2_label = np.full((merged_feature_matrix2_len,), 2)
feature_matrix2 = np.hstack((merged_feature_matrix2, task2_label.reshape(-1, 1)))

merged_feature_matrix3_len = len(merged_feature_matrix3)
task3_label = np.full((merged_feature_matrix3_len,), 3)
feature_matrix3 = np.hstack((merged_feature_matrix3, task3_label.reshape(-1, 1)))

merged_feature_matrix4_len = len(merged_feature_matrix4)
task4_label = np.full((merged_feature_matrix4_len,), 4)
feature_matrix4 = np.hstack((merged_feature_matrix4, task4_label.reshape(-1, 1)))

merged_feature_matrix = np.vstack((feature_matrix1,
                                   feature_matrix2,
                                   feature_matrix3,
                                   feature_matrix4))

normalized_features_min_max_with_label = normalize_columns(merged_feature_matrix)

print("归一化后的特征矩阵（包含标签，最小-最大缩放）：")
print(merged_feature_matrix)
print(normalized_features_min_max_with_label)
normalized_features_min_max_with_label = pd.DataFrame(normalized_features_min_max_with_label,
                                                      columns=['time_niu',

                                                               'time_median',
                                                               'time_max_min_diff',
                                                               'time_mad',
                                                               'time_variance',
                                                               'time_sigma',
                                                               'time_cv',
                                                               'time_skew',
                                                               'time_kurt',

                                                               'speed_niu',

                                                               'speed_median',
                                                               'speed_max_min_diff',
                                                               'speed_mad',
                                                               'speed_variance',
                                                               'speed_sigma',
                                                               'speed_cv',
                                                               'speed_skew',
                                                               'speed_kurt',

                                                               'distance_niu',

                                                               'distance_median',
                                                               'distance_max_min_diff',
                                                               'distance_mad',
                                                               'distance_variance',
                                                               'distance_sigma',
                                                               'distance_cv',
                                                               'distance_skew',
                                                               'distance_kurt',

                                                               'duration_len',
                                                               'speed_len',

                                                               'diameter_change_niu_0',

                                                               'diameter_change_median_0',
                                                               'diameter_change_max_min_diff_0',
                                                               'diameter_change_mad_0',
                                                               'diameter_change_variance_0',
                                                               'diameter_change_sigma_0',
                                                               'diameter_change_cv_0',
                                                               'diameter_change_skew_0',
                                                               'diameter_change_kurt_0',

                                                               'diameter_change_niu_1',

                                                               'diameter_change_median_1',
                                                               'diameter_change_max_min_diff_1',
                                                               'diameter_change_mad_1',
                                                               'diameter_change_variance_1',
                                                               'diameter_change_sigma_1',
                                                               'diameter_change_cv_1',
                                                               'diameter_change_skew_1',
                                                               'diameter_change_kurt_1',

                                                               'diameter_niu_0',

                                                               'diameter_median_0',
                                                               'diameter_max_min_diff_0',
                                                               'diameter_mad_0',
                                                               'diameter_variance_0',
                                                               'diameter_sigma_0',
                                                               'diameter_cv_0',
                                                               'diameter_skew_0',
                                                               'diameter_kurt_0',

                                                               'diameter_niu_1',

                                                               'diameter_median_1',
                                                               'diameter_max_min_diff_1',
                                                               'diameter_mad_1',
                                                               'diameter_variance_1',
                                                               'diameter_sigma_1',
                                                               'diameter_cv_1',
                                                               'diameter_skew_1',
                                                               'diameter_kurt_1',

                                                               'labels'
                                                               ])
features = normalized_features_min_max_with_label.drop(columns=['labels'])  # 特征矩阵
labels = normalized_features_min_max_with_label['labels']  # 标签
# 计算每个特征与标签之间的Pearson相关系数
correlation_coefficients = {}
for column in features.columns:
    correlation_coefficient = np.corrcoef(features[column], labels)[0, 1]
    correlation_coefficients[column] = correlation_coefficient

print("每个特征与标签的Pearson相关系数：")
for feature, correlation_coefficient in correlation_coefficients.items():
    print(f"{feature}: {correlation_coefficient}")

# 假设correlation_coefficients是一个字典，包含每个特征与标签的Pearson相关系数

# 将字典转换为DataFrame方便绘图
correlation_df = pd.DataFrame(list(correlation_coefficients.items()), columns=['Feature', 'Correlation'])

# 绘制散点图
plt.figure(figsize=(8, 6))

# 使用seaborn绘图


# 指定中文字体路径
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

sns.scatterplot(x='Feature', y='Correlation', data=correlation_df)
# plt.title('特征与标签的Pearson相关系数')
plt.xlabel('特征')
plt.ylabel('Pearson相关系数')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

df = pd.DataFrame.from_dict(correlation_coefficients, orient='index', columns=['Pearson Correlation'])

# 重新设置索引名，以在热力图上显示
df.index.name = 'Features'

# 使用Seaborn绘制热力图
plt.figure(figsize=(10, 8))  # 可以调整大小以适应你的需求
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
sns.heatmap(df, annot=True, fmt=".2f", cmap='coolwarm')  # annot=True表示显示数值，fmt控制数值的格式

# 设置标题和坐标轴标签
# plt.title('特征与标签的Pearson相关系数')
# plt.xlabel('Pearson相关系数')
plt.ylabel('特征')

# 显示图形
plt.show()

outputpath = 'OUT12.csv'
normalized_features_min_max_with_label.to_csv(outputpath, sep=',', index=False, header=True)
feature_columns = ['time_niu',

                   'time_median',
                   'time_max_min_diff',
                   'time_mad',
                   'time_variance',
                   'time_sigma',
                   'time_cv',
                   'time_skew',
                   'time_kurt',

                   'speed_niu',

                   'speed_median',
                   'speed_max_min_diff',
                   'speed_mad',
                   'speed_variance',
                   'speed_sigma',
                   'speed_cv',
                   'speed_skew',
                   'speed_kurt',

                   'distance_niu',

                   'distance_median',
                   'distance_max_min_diff',
                   'distance_mad',
                   'distance_variance',
                   'distance_sigma',
                   'distance_cv',
                   'distance_skew',
                   'distance_kurt',

                   'duration_len',
                   'speed_len',

                   'diameter_change_niu_0',

                   'diameter_change_median_0',
                   'diameter_change_max_min_diff_0',
                   'diameter_change_mad_0',
                   'diameter_change_variance_0',
                   'diameter_change_sigma_0',
                   'diameter_change_cv_0',
                   'diameter_change_skew_0',
                   'diameter_change_kurt_0',

                   'diameter_change_niu_1',

                   'diameter_change_median_1',
                   'diameter_change_max_min_diff_1',
                   'diameter_change_mad_1',
                   'diameter_change_variance_1',
                   'diameter_change_sigma_1',
                   'diameter_change_cv_1',
                   'diameter_change_skew_1',
                   'diameter_change_kurt_1',

                   'diameter_niu_0',

                   'diameter_median_0',
                   'diameter_max_min_diff_0',
                   'diameter_mad_0',
                   'diameter_variance_0',
                   'diameter_sigma_0',
                   'diameter_cv_0',
                   'diameter_skew_0',
                   'diameter_kurt_0',

                   'diameter_niu_1',

                   'diameter_median_1',
                   'diameter_max_min_diff_1',
                   'diameter_mad_1',
                   'diameter_variance_1',
                   'diameter_sigma_1',
                   'diameter_cv_1',
                   'diameter_skew_1',
                   'diameter_kurt_1', ]  # 这里添加所有特征列的名称
label_column = 'labels'  # 标签列的名称

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(normalized_features_min_max_with_label[feature_columns],
                                                    normalized_features_min_max_with_label[label_column],
                                                    test_size=0.26, random_state=42)

print('****计算完成***')
# print(merged_feature_matrix)
# print(merged_feature_matrix_test)


# svm训练
accuracy = train_and_predict_svm(X_train, y_train, X_test, y_test)
print('准确率：', accuracy)
features_matrix = X_train
labels = y_train
# 使用随机森林算法进行分类
classifier = ra_tree.random_forest_classifier_matrix(features_matrix, labels)

# 打印模型信息
print("训练好的随机森林分类器模型：\n", classifier)

# 在测试集上进行预测
y_pred = classifier.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("模型在测试集上的准确率：", accuracy)
precision = precision_score(y_test, y_pred, average='macro')
print("Precision:", precision)

# 计算召回率
recall = recall_score(y_test, y_pred, average='weighted')
print("Recall:", recall)

svm_model, test_accuracy = train_svm_classifier(X_train, y_train, X_test, y_test,
                                                kernel='linear',
                                                C=1.0,
                                                random_state=None,
                                                plot_decision_boundary=True,
                                                plot_learning_curve=False,
                                                plot_validation_curve=False,
                                                plot_support_vectors=False)
print("Test Accuracy:", test_accuracy)
