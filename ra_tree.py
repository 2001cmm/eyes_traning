import numpy as np
from sklearn.ensemble import RandomForestClassifier


def random_forest_classifier_matrix(features_matrix, labels, n_estimators=100, max_depth=None, random_state=42):
    """
    使用随机森林算法对输入为矩阵的数据进行分类。

    参数：
    features_matrix：特征矩阵，每行代表一个样本，每列代表一个特征。
    labels：标签向量，每个元素代表相应样本的类别。
    n_estimators：随机森林中树的数量，默认为100。
    max_depth：每棵树的最大深度，默认为None。
    random_state：随机种子，默认为42。

    返回：
    classifier：训练好的随机森林分类器模型。
    """
    # 初始化随机森林分类器
    classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)

    # 训练模型
    classifier.fit(features_matrix, labels)

    return classifier


# # 示例数据
# features_matrix = np.array([[1, 2, 3, 4],
#                             [5, 6, 7, 8],
#                             [9, 10, 11, 12],
#                             [13, 14, 15, 16],
#                             [17, 18, 19, 20],
#                             [21, 22, 23, 24],
#                             [25, 26, 27, 28],
#                             [29, 30, 31, 32]])
# labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
#
# # 使用随机森林算法进行分类
# classifier = random_forest_classifier_matrix(features_matrix, labels)
#
# # 打印模型信息
# print("训练好的随机森林分类器模型：\n", classifier)
