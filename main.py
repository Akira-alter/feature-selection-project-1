"""
实验入口与引擎。
核心优化：selector 每 (repeat, unlabeled_rate, method) 只 fit 一次，
不同 feature_ratio 直接从排名切片，拟合次数减少 ~9 倍。
"""
import time
import numpy as np

import config as cfg
from data_loader import load_data, preprocess
from splitter import generate_seeds, split_train_test, split_labeled_unlabeled
from classifiers import build_classifiers, evaluate
from results_manager import ResultsManager

class ProposedFeatureSelector:
    def fit(self, X, y):
        raise NotImplementedError(
            "The proposed algorithm will be released after paper acceptance."
        )
# ═══════════════════════ 算法注册 ═══════════════════════
# 新增算法只需: 1) import  2) 在此字典加一行  3) config.py 中声明监督类型
from otherAlgorithms import  SCFS, FMIUFS, RRPC
from OA_2 import SSNDI, SemiFREE_fast
from SemiDPAI import SemiDP, SemiAI
from T3I import T3I
from IMP1ARA import IMP1ARA
from FBIGCSFS import FBIGCSFS



METHODS = {
    'ProposedFeatureSelector':     ProposedFeatureSelector,
    # 'RRPC':     RRPC,           #17
    'SemiFREE': SemiFREE_fast,  #23
    'SSNDI':    SSNDI,          #20
    'SCFS':     SCFS,           #20
    'FMIUFS':   FMIUFS,         #21
    'SemiDP':   SemiDP,         #23
    'SemiAI':   SemiAI,         #23
    'T3I':      T3I,            #24
    'IMP1ARA':  IMP1ARA,        #23
    'FBIGCSFS': FBIGCSFS,       #25
    # 'NewAlgo': NewAlgo,   ← 接入新算法
}


def _get_supervision_type(name: str) -> str:
    if name in cfg.UNSUPERVISED_METHODS:
        return "unsupervised"
    if name in cfg.SUPERVISED_METHODS:
        return "supervised"
    return "semi-supervised"


# def _fit_and_rank(name, cls, X_lab, y_lab, X_unl, n_features, sup_type):
#     """
#     拟合一次，返回完整特征排名索引（长度 = n_features，按重要性降序）。
#     这是本次重构的 *最关键优化点*。
#     """
#     selector = cls(n_selected_features=n_features)
#
#     if sup_type == "unsupervised":
#         selector.fit(X_unl)
#     elif sup_type == "supervised":
#         selector.fit(X_lab, y_lab)
#     else:
#         selector.fit(X_lab, y_lab, X_unl)
#
#     ranking = selector.transform(X_lab).astype(int)
#     return ranking


def _try_get_ranking(selector_cls, sup_type, X_lab, y_lab, X_unl, n_features):
    """
    尝试拿到完整排名。如果 selector 有 scores_ 或 ranking_ 属性则利用，
    否则返回 None，调用方回退到逐 ratio fit。
    """
    selector = selector_cls(n_selected_features=n_features)
    if sup_type == "unsupervised":
        selector.fit(X_unl)
    elif sup_type == "supervised":
        selector.fit(X_lab, y_lab)
    else:
        selector.fit(X_lab, y_lab, X_unl)

    # 优先用 scores_/ranking_ 属性（如果算法暴露了的话）
    if hasattr(selector, 'scores_'):
        return np.argsort(selector.scores_)[::-1]  # 降序
    if hasattr(selector, 'ranking_'):
        return np.argsort(selector.ranking_)        # ranking_ 越小越重要
    if hasattr(selector, 'feature_importances_'):
        return np.argsort(selector.feature_importances_)[::-1]

    return None  # 无法确定排名顺序，必须逐 ratio fit


def _fit_select(selector_cls, sup_type, X_lab, y_lab, X_unl, k):
    """安全回退：按指定 k 独立 fit + select。"""
    selector = selector_cls(n_selected_features=k)
    if sup_type == "unsupervised":
        selector.fit(X_unl)
    elif sup_type == "supervised":
        selector.fit(X_lab, y_lab)
    else:
        selector.fit(X_lab, y_lab, X_unl)
    return selector.transform(X_lab).astype(int)

# ═══════════════════════ 实验主循环 ═══════════════════════

def run_all():
    results = ResultsManager()

    # 预生成全部种子 → 可复现
    seeds = generate_seeds(cfg.BASE_SEED, cfg.N_REPEATS * 2)
    split_seeds = seeds[: cfg.N_REPEATS]
    label_seeds = seeds[cfg.N_REPEATS :]

    for ds_cfg in cfg.DATASETS:
        ds_name = ds_cfg["name"]
        print(f"\n{'=' * 60}\n  Dataset: {ds_name}\n{'=' * 60}")

        X, y = load_data(ds_cfg, cfg.DATA_DIR)
        X, y = preprocess(X, y)
        n_feat = X.shape[1]

        # 每个数据集只建一次分类器对象（fit 会覆盖旧状态）
        classifiers = build_classifiers(cfg.CLASSIFIER_PARAMS)

        n_total = (cfg.N_REPEATS * len(cfg.UNLABELED_RATES)
                   * len(METHODS) * len(cfg.FEATURE_RATIOS)
                   * len(classifiers))
        done = 0
        t0 = time.time()

        for rep in range(cfg.N_REPEATS):
            X_tr, X_te, y_tr, y_te = split_train_test(
                X, y, cfg.TEST_SIZE, split_seeds[rep])

            for ur in cfg.UNLABELED_RATES:
                # 种子与 unlabeled_rate 耦合，保证不同 rate 用不同划分
                lu_seed = label_seeds[rep] + round(ur * 1000)
                X_lab, X_unl, y_lab, _ = split_labeled_unlabeled(
                    X_tr, y_tr, ur, lu_seed)

                for m_name, m_cls in METHODS.items():
                    sup = _get_supervision_type(m_name)

                    # 先尝试一次 fit 拿排名
                    ranking = _try_get_ranking(m_cls, sup, X_lab, y_lab, X_unl, n_feat)

                    # ★★★ 只 fit 一次，拿到完整排名 ★★★
                    # try:
                    #     ranking = _fit_and_rank(
                    #         m_name, m_cls,
                    #         X_lab, y_lab, X_unl, n_feat, sup)
                    # except Exception as e:
                    #     print(f"\n  [WARN] {m_name} fit failed: {e}")
                    #     done += len(cfg.FEATURE_RATIOS) * len(classifiers)
                    #     continue

                    for fr in cfg.FEATURE_RATIOS:
                        k = max(1, int(n_feat * fr))

                        if ranking is not None:
                            sel_idx = ranking[:k]
                        else:
                            # 回退：独立 fit
                            sel_idx = _fit_select(m_cls, sup, X_lab, y_lab, X_unl, k)

                        # sel_idx = ranking[:k]
                        if len(sel_idx) == 0:
                            done += len(classifiers)
                            continue

                        X_tr_s = X_tr[:, sel_idx]
                        X_te_s = X_te[:, sel_idx]

                        for c_name, clf in classifiers.items():
                            acc = evaluate(clf, X_tr_s, y_tr, X_te_s, y_te)
                            results.add(
                                dataset=ds_name,
                                method=m_name,
                                classifier=c_name,
                                unlabeled_rate=ur,
                                feature_ratio=fr,
                                repeat=rep,
                                accuracy=acc,
                            )
                            done += 1

                    # 进度条
                    pct = done / n_total * 100
                    elapsed = time.time() - t0
                    print(f"\r  [{ds_name}] {done}/{n_total} "
                          f"({pct:.1f}%) | {elapsed:.0f}s",
                          end="", flush=True)

        print(f"\n  ✓ {ds_name} done in {time.time() - t0:.1f}s")

    return results


# ═══════════════════════ 入口 ═══════════════════════

def main():
    print("=" * 60)
    print("  Semi-Supervised Feature Selection Experiment")
    print("=" * 60)

    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg.FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    results = run_all()

    # ── 保存 CSV ──
    print("\n--- Saving CSV ---")
    results.save_summary_csv(cfg.OUTPUT_DIR)
    results.save_detailed_csv(cfg.OUTPUT_DIR)
    # 原始全量数据备份
    raw = cfg.OUTPUT_DIR / "raw_results.csv"
    results.to_dataframe().to_csv(raw, index=False, float_format="%.6f")
    print(f"  [CSV] {raw}")

    # ── 绘图 ──
    print("\n--- Generating Figures ---")
    results.plot_acc_vs_unlabeled_rate(cfg.FIGURE_DIR, cfg.FIG_FORMAT, cfg.FIG_DPI)
    results.plot_acc_vs_feature_ratio(cfg.FIGURE_DIR, cfg.FIG_FORMAT, cfg.FIG_DPI)

    print("\n✓ All experiments completed.")


if __name__ == "__main__":
    main()