import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import learning_curve, validation_curve


def train_svm_classifier(X_train, y_train, X_test, y_test, kernel='linear', C=1.0, random_state=None,
                         plot_decision_boundary=False, plot_learning_curve=False, plot_validation_curve=False,
                         plot_support_vectors=False):
    """
    使用 SVM 训练一个分类器，并返回训练好的模型、在测试集上的准确度，以及可选的图形。

    参数：
    X_train: array-like, shape (n_samples, n_features)
        训练数据集的特征向量。
    y_train: array-like, shape (n_samples,)
        训练数据集的目标标签。
    X_test: array-like, shape (n_samples, n_features)
        测试数据集的特征向量。
    y_test: array-like, shape (n_samples,)
        测试数据集的目标标签。
    kernel: str, optional (default='linear')
        SVM 的核函数。可选值包括 'linear', 'poly', 'rbf', 'sigmoid' 等。
    C: float, optional (default=1.0)
        SVM 的惩罚参数。
    random_state: int, RandomState instance or None, optional (default=None)
        随机数种子。
    plot_decision_boundary: bool, optional (default=False)
        是否绘制决策边界可视化图。
    plot_learning_curve: bool, optional (default=False)
        是否绘制学习曲线。
    plot_validation_curve: bool, optional (default=False)
        是否绘制超参数调优曲线。
    plot_support_vectors: bool, optional (default=False)
        是否绘制支持向量可视化图。

    返回：
    model: SVM 模型
        训练好的 SVM 模型。
    accuracy: float
        模型在测试集上的准确度。
    """
    # 创建 SVM 分类器
    svm_classifier = svm.SVC(kernel=kernel, C=C, random_state=random_state)

    # 训练 SVM 分类器
    svm_classifier.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = svm_classifier.predict(X_test)

    # 计算准确度
    accuracy = accuracy_score(y_test, y_pred)

    # 输出分类报告和混淆矩阵
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # 绘制决策边界可视化图
    if plot_decision_boundary:
        plt.figure(figsize=(8, 6))
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm, s=30)
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xx = np.linspace(xlim[0], xlim[1], 30)
        yy = np.linspace(ylim[0], ylim[1], 30)
        XX, YY = np.meshgrid(xx, yy)
        if X_train.shape == X_test.shape:
            Z = svm_classifier.decision_function(np.c_[XX.ravel(), YY.ravel()])
            Z = Z.reshape(XX.shape)
            ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
            plt.title('Decision Boundary of SVM Classifier')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.show()

    # 绘制学习曲线
    if plot_learning_curve:
        plt.figure(figsize=(8, 6))
        train_sizes, train_scores, valid_scores = learning_curve(svm_classifier, X_train, y_train,
                                                                 train_sizes=np.linspace(0.1, 1.0, 10), cv=5)
        plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training score')
        plt.plot(train_sizes, np.mean(valid_scores, axis=1), 'o-', label='Cross-validation score')
        plt.xlabel('Training examples')
        plt.ylabel('Score')
        plt.title('Learning Curve of SVM Classifier')
        plt.legend(loc='best')
        plt.show()

    # 绘制超参数调优曲线
    if plot_validation_curve:
        plt.figure(figsize=(8, 6))
        param_range = np.logspace(-3, 3, 6)
        train_scores, valid_scores = validation_curve(svm_classifier, X_train, y_train, param_name="C",
                                                      param_range=param_range, cv=5)
        plt.plot(param_range, np.mean(train_scores, axis=1), 'o-', label='Training score')
        plt.plot(param_range, np.mean(valid_scores, axis=1), 'o-', label='Cross-validation score')
        plt.xscale('log')
        plt.xlabel('Parameter C')
        plt.ylabel('Score')
        plt.title('Validation Curve of SVM Classifier')
        plt.legend(loc='best')
        plt.show()

    # 绘制支持向量可视化图
    if plot_support_vectors:
        plt.figure(figsize=(8, 6))
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm, s=30)
        plt.scatter(svm_classifier.support_vectors_[:, 0], svm_classifier.support_vectors_[:, 1], s=100,
                    facecolors='none', edgecolors='k')
        plt.title('Support Vectors of SVM Classifier')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()

    return svm_classifier, accuracy
