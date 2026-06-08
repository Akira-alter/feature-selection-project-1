"""
实验配置：所有参数集中管理，修改参数只改这一个文件。
"""
from pathlib import Path

# 当前 config.py 文件所在目录
BASE_DIR = Path(__file__).resolve().parent

# ═══════════════════════ 路径 ═══════════════════════
DATA_DIR   = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "results"
FIGURE_DIR = BASE_DIR / "figures"


# ═══════════════════════ 并行与缓存参数 ═══════════════════════
CACHE_DIR = BASE_DIR / "cache"
SELECTOR_CACHE_DIR = CACHE_DIR / "selectors"
RESULT_SHARD_DIR = CACHE_DIR / "result_shards"

USE_CACHE = True

# 如果你改了特征选择算法源码，建议临时改成 True，强制重算 selector
FORCE_RECOMPUTE_SELECTOR = False

# 如果你改了分类器参数、feature ratios、实验设置，建议临时改成 True，强制重算结果分片
FORCE_RECOMPUTE_RESULTS = False

# 并行进程数。不要一开始就用 -1，容易内存爆。
PARALLEL_N_JOBS = 6

# joblib 会把大数组自动 memmap，减少多进程复制内存
JOBLIB_MAX_NBYTES = "64M"

# 缓存版本号。你大改代码后，可以改成 "v2" 让旧缓存自动失效。
CACHE_VERSION = "v1"

# ═══════════════════════ 数据集注册表 ═══════════════════════
# 新增数据集只需在此追加一条记录
# format: csv / txt / mat / arff
# feature_cols: slice 或 list；label_col: int（列索引）或 str（列名）
DATASETS = [

    
    #Rice缓存在v2里面了
    {
       "name": "Rice",
       "file": "Rice_Cammeo_Osmancik.arff",
       "format": "arff",
       "feature_cols": slice(0, 7),
       "label_col": 7,
    },


        {
        "name": "ionosphere",
        "file": "ionosphere.data",
        "format": "csv",
        "feature_cols": slice(0, 34),
        "label_col": 34,
    },

    {
        "name": "sonar",
        "file": "sonar.all-data",
        "format": "csv",
        "feature_cols": slice(0, 60),
        "label_col": 60,
    },

    {
        "name": "occupancy",
        "file": "datatest.txt",
        "format": "csv",
        "sep": ",",
        "header": 0,
        "feature_cols": slice(1, 6),
        "label_col": 6,
    },

  

    # 继续添加更多数据集 ...
]

# ═══════════════════════ 实验参数 ═══════════════════════
N_REPEATS       = 10                                           # 重复次数
TEST_SIZE       = 0.8                                          # 80% 测试 / 20% 训练
UNLABELED_RATES = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]        # 未标记率
FEATURE_RATIOS  = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
BASE_SEED       = 42                                           # 可复现种子基数

# ═══════════════════════ 分类器参数 ═══════════════════════
CLASSIFIER_PARAMS = {
    "KNN":  {"n_neighbors": 5},
    "SVM":  {"C": 1.0, "class_weight": "balanced", "max_iter": 5000},
    "CART": {"criterion": "gini", "max_depth": None},
}

# ═══════════════════════ 特征选择方法监督类型 ═══════════════════════
# 新增对比算法时，若不在以下两集合中，默认按半监督处理
UNSUPERVISED_METHODS = {"SCFS", "FMIUFS"}
SUPERVISED_METHODS   = {"mRMR"}

# ═══════════════════════ 图表参数 ═══════════════════════
FIG_FORMAT = "png"      # pdf / png / svg
FIG_DPI    = 300