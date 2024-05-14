import numpy as np
import math

def calc(data):
    n = len(data)
    niu = sum(data) / n  # 平均值
    niu2 = sum([x ** 2 for x in data]) / n  # 平方的平均值
    niu3 = sum([x ** 3 for x in data]) / n  # 三次方的平均值
    niu4 = sum([x ** 4 for x in data]) / n  # 四次方的平均值
    sigma = math.sqrt(niu2 - niu * niu)  # 标准差
    return [niu, sigma, niu2, niu3, niu4]

def calc_stat(data):
    # 确保数据是数值类型
    if not all(isinstance(x, (int, float)) for x in data):
        raise ValueError("Data must contain only numerical values.")
    # 调用 calc 函数计算基本统计量
    [niu, sigma, niu2, niu3, niu4] = calc(data)
    n = len(data)
    # 转换为 NumPy 数组以进行高效计算
    data_np = np.array(data)
    # 中位数
    median = np.median(data_np)
    # 极差 (最大值与最小值的差)
    max_min_diff = np.max(data_np) - np.min(data_np)
    # 平均绝对偏差 (MAD)
    mad = np.mean(np.abs(data_np - niu))
    # 方差
    variance = niu2 - niu * niu
    # 变异系数 (Coefficient of Variation, CV)，以百分比表示
    cv = (sigma / niu) * 100 if niu != 0 else 0
    # 偏度
    skew = 0 if n < 3 else (niu3 - 3 * niu * sigma ** 2 - niu ** 3) / (sigma ** 3)
    # 峰度
    kurt = 0 if n < 4 else (niu4 - 4 * niu * niu3 + 3 * niu ** 2 * niu) / (sigma ** 4)
    return [
        niu, median, max_min_diff, mad, variance, sigma, cv, skew, kurt
    ]

# # 示例用法
# data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 您的特征数据
# selected_features = calc_stat(data)
# print(selected_features)
