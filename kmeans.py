import numpy as np
from sklearn.cluster import KMeans


def cluster_1d_data(data, n_clusters):
    """
    使用 K-Means 算法对一维数据进行聚类，并返回聚类结果。

    参数：
    data: list
        包含一维数据的列表。
    n_clusters: int
        聚类簇的数量。

    返回：
    list
        包含每个数据点所属聚类簇的标签。
    """
    # 将一维数据转换成 N × 1 的矩阵
    X = np.array(data).reshape(-1, 1)

    # 创建 K-Means 模型并进行聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)

    # 返回每个数据点所属的聚类簇标签
    return kmeans.labels_


# 示例数据
# data = [2, 3, 6, 8, 10, 12, 15, 18, 20]
# n_clusters = 3
#
# # 调用函数进行聚类
# cluster_labels = cluster_1d_data(data, n_clusters)
#
# # 输出聚类结果
# print("聚类标签：", cluster_labels)
