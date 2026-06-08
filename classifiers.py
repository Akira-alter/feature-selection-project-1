"""
分类器统一封装：构建 → 训练 → 预测 → 评估一步完成。
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


_CLF_MAP = {
    "KNN":  KNeighborsClassifier,
    "SVM":  LinearSVC,
    "CART": DecisionTreeClassifier,
}


def build_classifiers(params: dict) -> dict:
    """根据配置一次性构建所有分类器实例（可重复 fit）。"""
    return {name: _CLF_MAP[name](**params.get(name, {}))
            for name in params}


def evaluate(clf, X_train, y_train, X_test, y_test) -> float:
    """训练 + 预测 + 返回准确率。clf 可复用，fit 会覆盖前次状态。"""
    clf.fit(X_train, y_train)
    return accuracy_score(y_test, clf.predict(X_test))


