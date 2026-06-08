"""
数据划分：训练/测试 + 已标记/未标记，种子集中管理保证可复现。
"""
import numpy as np
from sklearn.model_selection import train_test_split
import warnings

def generate_seeds(base_seed: int, n: int) -> list:
    """从基础种子确定性地生成 n 个子种子。"""
    rng = np.random.RandomState(base_seed)
    return rng.randint(0, 2**31, size=n).tolist()


def split_train_test(X, y, test_size: float, seed: int):
    """80% 测试 / 20% 训练（按 request 要求）。"""
    try:
        return train_test_split(X, y, test_size=test_size,
                                random_state=seed, stratify=y)
    except ValueError:
        # # 类别极少时 stratify 可能失败
        # warnings.warn(
        #     f"Stratified split failed (seed={seed}, size={test_size}): {e}. "
        #     f"Falling back to non-stratified split.",
        #     RuntimeWarning
        # )
        return train_test_split(X, y, test_size=test_size,
                                random_state=seed)


def split_labeled_unlabeled(X_train, y_train, unlabeled_rate: float, seed: int):
    """
    从训练集中按 unlabeled_rate 划分。
    返回 (X_labeled, X_unlabeled, y_labeled, y_unlabeled)
    """
    labeled_rate = 1.0 - unlabeled_rate
    try:
        return train_test_split(X_train, y_train,
                                train_size=labeled_rate,
                                random_state=seed, stratify=y_train)
    except ValueError:

        return train_test_split(X_train, y_train,
                                train_size=labeled_rate,
                                random_state=seed)