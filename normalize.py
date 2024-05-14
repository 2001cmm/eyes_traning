import numpy as np

def normalize_features(features, method='min-max', include_label=False):
    """
    对特征矩阵进行归一化

    参数：
        features: numpy数组，形状为 (n, m)，表示特征矩阵，n 为样本数量，m 为特征数量
        method: 字符串，表示归一化方法，可选值为 'min-max' 或 'standard'，默认为 'min-max'
        include_label: 布尔值，表示是否包含最后一列标签，默认为 False

    返回：
        normalized_features: numpy数组，形状与输入 features 相同，表示归一化后的特征矩阵
    """
    if include_label:
        feature_matrix = features[:, :-1]  # 排除最后一列标签
    else:
        feature_matrix = features

    if method == 'min-max':
        # 最小-最大缩放
        min_vals = np.min(feature_matrix, axis=0)
        max_vals = np.max(feature_matrix, axis=0)
        normalized_features = (feature_matrix - min_vals) / (max_vals - min_vals)
    elif method == 'standard':
        # 标准化
        mean_vals = np.mean(feature_matrix, axis=0)
        std_vals = np.std(feature_matrix, axis=0)
        normalized_features = (feature_matrix - mean_vals) / std_vals
    else:
        raise ValueError("Invalid normalization method. Choose 'min-max' or 'standard'.")

    if include_label:
        normalized_features = np.hstack((normalized_features, features[:, -1].reshape(-1, 1)))  # 添加回最后一列标签

    return normalized_features

# # 测试函数
# # 创建示例特征矩阵（包含标签）
# features_with_label = np.array([
#     [1, 2, 3, 10],
#     [4, 5, 6, 20],
#     [7, 8, 9, 30],
#     [10, 11, 12, 40],
#     [13, 14, 15, 50]
# ])
#
# # 对特征矩阵进行归一化（最小-最大缩放），包含最后一列标签
# normalized_features_min_max_with_label = normalize_features(features_with_label, method='min-max', include_label=True)
# print("归一化后的特征矩阵（包含标签，最小-最大缩放）：")
# print(normalized_features_min_max_with_label)
#
# # 对特征矩阵进行归一化（标准化），不包含最后一列标签
# normalized_features_standard_without_label = normalize_features(features_with_label, method='standard', include_label=False)
# print("归一化后的特征矩阵（不包含标签，标准化）：")
# print(normalized_features_standard_without_label)
