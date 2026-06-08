"""
数据加载与预处理：支持 csv / txt / mat / arff，输出统一 numpy 结构。
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder


def _process_mixed_feature_df(X_df: pd.DataFrame, numeric_threshold: float = 0.8):
    """
    对包含混合类型特征的 DataFrame 做稳健处理：
    1) "?", "", "NA", "null" 等记为缺失值
    2) 若一列大部分值可转为数值，则按数值列处理
    3) 否则按类别列做整数编码
    返回:
        X: ndarray[float64]
    """
    X_df = X_df.copy()

    # 常见缺失值标记
    missing_tokens = {"?", "", "NA", "N/A", "na", "null", "NULL", "None", "none"}

    for col in X_df.columns:
        # 统一把字节串转成字符串（主要兼容 arff 里可能出现的 bytes）
        X_df[col] = X_df[col].apply(
            lambda v: v.decode("utf-8", errors="ignore") if isinstance(v, (bytes, bytearray)) else v
        )

        # 先把 object 列中的常见缺失标记替换为 np.nan
        X_df[col] = X_df[col].apply(
            lambda v: np.nan if (pd.isna(v) or (isinstance(v, str) and v.strip() in missing_tokens)) else v
        )

        # 尝试转成数值
        numeric_col = pd.to_numeric(X_df[col], errors="coerce")
        valid_ratio = numeric_col.notna().mean()

        # 大多数可转成数值：按数值列处理
        if valid_ratio >= numeric_threshold:
            X_df[col] = numeric_col.astype(np.float64)
        else:
            # 按类别列处理：缺失值作为单独类别 MISSING
            cat_col = X_df[col].astype("object")
            cat_col = cat_col.where(~pd.isna(cat_col), "MISSING")
            cat_col = cat_col.astype(str)

            codes, _ = pd.factorize(cat_col)
            X_df[col] = codes.astype(np.float64)

    return X_df.to_numpy(dtype=np.float64)


def load_data(ds_cfg: dict, data_dir: Path):
    """
    根据配置加载数据集，返回 (X: ndarray[float64], y: ndarray)。
    支持混合类型特征：
    - 数值列保留为 float64
    - 非数值列自动编码为整数
    - "?" 等标记视为缺失值
    """
    file_path = data_dir / ds_cfg["file"]
    fmt = ds_cfg.get("format", "csv")



    if fmt in ("csv", "txt"):
        sep = ds_cfg.get("sep", ",")
        header = ds_cfg.get("header", 0)

        # 支持空格 / 多个空格 / 制表符分隔
        # 如果配置里写 sep: "space" 或 sep: "whitespace"，就按任意连续空白读取
        if sep in ("space", "whitespace", r"\s+"):
            sep = r"\s+"

        data = pd.read_csv(
            file_path,
            sep=sep,
            header=header,
            engine="python" if sep == r"\s+" else "c",
            na_values=["?", "", "NA", "N/A", "na", "null", "NULL", "None", "none"]
        )

        X_df = data.iloc[:, ds_cfg["feature_cols"]].copy()
        X = _process_mixed_feature_df(X_df)
        y = data.iloc[:, ds_cfg["label_col"]].values

    elif fmt == "mat":
        from scipy.io import loadmat
        mat = loadmat(str(file_path))
        X = np.asarray(mat[ds_cfg["feature_key"]], dtype=np.float64)
        y = np.asarray(mat[ds_cfg["label_key"]]).ravel()

    elif fmt == "arff":
        from scipy.io import arff as arff_io
        raw, _ = arff_io.loadarff(str(file_path))
        df = pd.DataFrame(raw)

        X_df = df.iloc[:, ds_cfg["feature_cols"]].copy()
        X = _process_mixed_feature_df(X_df)
        y = df.iloc[:, ds_cfg["label_col"]].values

    else:
        raise ValueError(f"Unsupported format: {fmt}")

    return X, y


def preprocess(X: np.ndarray, y: np.ndarray):
    """
    1) 缺失值 → 列均值填充
    2) Min-Max 归一化到 [0, 1]
    3) 标签 → 整数编码
    """
    X = X.astype(np.float64, copy=False)

    # ── 缺失值 ──
    nan_mask = np.isnan(X)
    if nan_mask.any():
        X = X.copy()
        col_mean = np.nanmean(X, axis=0)

        # 如果某一列全是 NaN，则 np.nanmean 会得到 NaN，这里兜底设为 0
        col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)

        for c in range(X.shape[1]):
            X[nan_mask[:, c], c] = col_mean[c]

    # ── 归一化 ──
    x_min = X.min(axis=0)
    x_range = X.max(axis=0) - x_min
    x_range[x_range == 0] = 1.0          # 常量列不缩放
    X = (X - x_min) / x_range

    # ── 标签编码 ──
    # 兼容 bytes / str / 数值标签
    y = np.array([
        v.decode("utf-8", errors="ignore") if isinstance(v, (bytes, bytearray)) else v
        for v in y
    ], dtype=object)

    le = LabelEncoder()
    y = le.fit_transform(y.astype(str))

    return X, y