import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def train_and_predict_svm(X_train, y_train, X_test, y_test, kernel='linear', C=1.0):
    """
    使用SVM进行训练和预测的函数

    参数：
    X_train：训练集特征矩阵
    y_train：训练集标签
    X_test：测试集特征矩阵
    y_test：测试集标签
    kernel：核函数，默认为线性核
    C：SVM正则化参数，默认为1.0

    返回值：
    accuracy：模型在测试集上的准确率
    """
    # # 确保训练集和测试集特征维度一致
    # if X_train.shape[1] != X_test.shape[1]:
    #     raise ValueError("训练集和测试集的特征维度不一致！")

    # 特征归一化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 初始化SVM模型
    svm_model = SVC(kernel=kernel, C=C, random_state=42)

    # 在训练集上训练模型
    svm_model.fit(X_train_scaled, y_train)

    # 在测试集上进行预测
    y_pred = svm_model.predict(X_test_scaled)

    # 计算模型准确率
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy


# 示例用法
# # 定义训练集特征矩阵及标签
# X_train = np.array([[2, 3], [4, 5], [6, 7], [8, 9]])
# y_train = np.array([0, 1, 0, 1])
#
# # 定义测试集特征矩阵及标签（调整特征维度）
# X_test = np.array([[1, 2, 3], [3, 4, 5]])  # 调整为与训练集相同的特征维度
# y_test = np.array([0, 1])
#
# # 调用函数进行训练和预测
# accuracy = train_and_predict_svm(X_train, y_train, X_test, y_test)
#
# print("模型准确率：", accuracy)
