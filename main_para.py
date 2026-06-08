


"""
并行版实验入口。

特点：
1. 按 (dataset, repeat, unlabeled_rate, method) 粒度并行；
2. 每个并行任务内部复用 ranking，避免不同 feature_ratio 重复 fit；
3. selector ranking / selected features 落盘缓存；
4. 每个任务完成后保存 result shard CSV；
5. 程序中断后可断点续跑；
6. 打印详细耗时，便于定位慢在 selector 还是 classifier。
"""

import json
import time
import hashlib
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone

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

from otherAlgorithms import SCFS, FMIUFS, RRPC
from OA_2 import SSNDI, SemiFREE_fast
from SemiDPAI import SemiDP, SemiAI
from T3I import T3I
from IMP1ARA import IMP1ARA
from FBIGCSFS import FBIGCSFS


# ═══════════════════════ 算法注册 ═══════════════════════
METHODS = {
    "ProposedFeatureSelector":     ProposedFeatureSelector,
    # "RRPC":     RRPC,
    # "SemiFREE": SemiFREE_fast,
    "SSNDI":    SSNDI,
    "SCFS":     SCFS,
    "FMIUFS":   FMIUFS,
    "SemiDP":   SemiDP,
    "SemiAI":   SemiAI,
    "T3I":      T3I,
    "IMP1ARA":  IMP1ARA,
    "FBIGCSFS": FBIGCSFS,
}


# ═══════════════════════ 基础工具函数 ═══════════════════════

def _get_supervision_type(name: str) -> str:
    if name in cfg.UNSUPERVISED_METHODS:
        return "unsupervised"
    if name in cfg.SUPERVISED_METHODS:
        return "supervised"
    return "semi-supervised"


def _safe_name(s: str) -> str:
    return str(s).replace(" ", "_").replace("/", "_").replace("\\", "_")


def _ensure_dirs():
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg.FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    cache_dir = getattr(cfg, "CACHE_DIR", cfg.BASE_DIR / "cache")
    selector_cache_dir = getattr(cfg, "SELECTOR_CACHE_DIR", cache_dir / "selectors")
    result_shard_dir = getattr(cfg, "RESULT_SHARD_DIR", cache_dir / "result_shards")

    cache_dir.mkdir(parents=True, exist_ok=True)
    selector_cache_dir.mkdir(parents=True, exist_ok=True)
    result_shard_dir.mkdir(parents=True, exist_ok=True)


def _hash_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _json_dumps_stable(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)


def _experiment_signature() -> str:
    """
    把影响结果的关键实验参数放进签名。
    如果这些参数变了，结果分片文件名也会变，避免误用旧结果。
    """
    payload = {
        "cache_version": getattr(cfg, "CACHE_VERSION", "v1"),
        "test_size": cfg.TEST_SIZE,
        "unlabeled_rates": cfg.UNLABELED_RATES,
        "feature_ratios": cfg.FEATURE_RATIOS,
        "base_seed": cfg.BASE_SEED,
        "classifier_params": cfg.CLASSIFIER_PARAMS,
        "unsupervised_methods": sorted(list(cfg.UNSUPERVISED_METHODS)),
        "supervised_methods": sorted(list(cfg.SUPERVISED_METHODS)),
    }
    return _hash_text(_json_dumps_stable(payload))


def _selector_cache_path(
    ds_name: str,
    method_name: str,
    rep: int,
    unlabeled_rate: float,
    split_seed: int,
    label_seed: int,
    n_features: int,
    kind: str,
    k: int | None = None,
) -> Path:
    """
    kind:
      - ranking: 完整特征排序
      - selected: 某个 k 对应的特征子集
    """
    selector_cache_dir = getattr(
        cfg,
        "SELECTOR_CACHE_DIR",
        getattr(cfg, "CACHE_DIR", cfg.BASE_DIR / "cache") / "selectors",
    )

    payload = {
        "cache_version": getattr(cfg, "CACHE_VERSION", "v1"),
        "dataset": ds_name,
        "method": method_name,
        "rep": rep,
        "unlabeled_rate": unlabeled_rate,
        "split_seed": split_seed,
        "label_seed": label_seed,
        "n_features": n_features,
        "kind": kind,
        "k": k,
    }
    digest = _hash_text(_json_dumps_stable(payload))

    filename = (
        f"{_safe_name(ds_name)}__{_safe_name(method_name)}__"
        f"rep{rep}__ur{unlabeled_rate}__{kind}"
    )

    if k is not None:
        filename += f"__k{k}"

    filename += f"__{digest}.npy"

    return selector_cache_dir / filename


def _result_shard_path(
    ds_name: str,
    method_name: str,
    rep: int,
    unlabeled_rate: float,
    split_seed: int,
    label_seed: int,
) -> Path:
    result_shard_dir = getattr(
        cfg,
        "RESULT_SHARD_DIR",
        getattr(cfg, "CACHE_DIR", cfg.BASE_DIR / "cache") / "result_shards",
    )

    payload = {
        "signature": _experiment_signature(),
        "dataset": ds_name,
        "method": method_name,
        "rep": rep,
        "unlabeled_rate": unlabeled_rate,
        "split_seed": split_seed,
        "label_seed": label_seed,
    }
    digest = _hash_text(_json_dumps_stable(payload))

    filename = (
        f"{_safe_name(ds_name)}__{_safe_name(method_name)}__"
        f"rep{rep}__ur{unlabeled_rate}__{digest}.csv"
    )

    return result_shard_dir / filename


def _save_csv_atomic(df: pd.DataFrame, path: Path):
    """
    先写临时文件，再 replace。
    避免程序中断时留下半个坏 CSV。
    """
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False, float_format="%.6f")
    tmp_path.replace(path)


# ═══════════════════════ selector 相关函数 ═══════════════════════

def _fit_selector(selector_cls, sup_type, X_lab, y_lab, X_unl, n_selected_features):
    selector = selector_cls(n_selected_features=n_selected_features)

    if sup_type == "unsupervised":
        selector.fit(X_unl)
    elif sup_type == "supervised":
        selector.fit(X_lab, y_lab)
    else:
        selector.fit(X_lab, y_lab, X_unl)

    return selector


def _ranking_from_selector(selector):
    """
    尝试从 selector 中提取完整特征排序。
    注意：
    scores_ / feature_importances_ 通常越大越重要；
    ranking_ 通常越小越重要。
    """
    if hasattr(selector, "scores_"):
        scores = np.asarray(selector.scores_)
        return np.argsort(scores)[::-1].astype(int)

    if hasattr(selector, "ranking_"):
        ranking = np.asarray(selector.ranking_)
        return np.argsort(ranking).astype(int)

    if hasattr(selector, "feature_importances_"):
        importances = np.asarray(selector.feature_importances_)
        return np.argsort(importances)[::-1].astype(int)

    return None


def _get_or_compute_ranking(
    ds_name,
    method_name,
    selector_cls,
    sup_type,
    X_lab,
    y_lab,
    X_unl,
    n_features,
    rep,
    unlabeled_rate,
    split_seed,
    label_seed,
):
    """
    优先读取完整 ranking 缓存。
    如果没有缓存，则 fit 一次 selector，并尝试提取 ranking。
    """
    use_cache = getattr(cfg, "USE_CACHE", True)
    force = getattr(cfg, "FORCE_RECOMPUTE_SELECTOR", False)

    path = _selector_cache_path(
        ds_name=ds_name,
        method_name=method_name,
        rep=rep,
        unlabeled_rate=unlabeled_rate,
        split_seed=split_seed,
        label_seed=label_seed,
        n_features=n_features,
        kind="ranking",
        k=None,
    )

    if use_cache and path.exists() and not force:
        t0 = time.time()
        ranking = np.load(path)
        return ranking.astype(int), {
            "selector_mode": "ranking_cache",
            "selector_time": time.time() - t0,
            "selector_cache_file": str(path),
        }

    t0 = time.time()
    selector = _fit_selector(
        selector_cls=selector_cls,
        sup_type=sup_type,
        X_lab=X_lab,
        y_lab=y_lab,
        X_unl=X_unl,
        n_selected_features=n_features,
    )
    ranking = _ranking_from_selector(selector)
    elapsed = time.time() - t0

    if ranking is None:
        return None, {
            "selector_mode": "no_ranking_after_fit",
            "selector_time": elapsed,
            "selector_cache_file": "",
        }

    ranking = np.asarray(ranking, dtype=int)

    if use_cache:
        np.save(path, ranking)

    return ranking, {
        "selector_mode": "ranking_fit",
        "selector_time": elapsed,
        "selector_cache_file": str(path),
    }


def _get_or_compute_selected(
    ds_name,
    method_name,
    selector_cls,
    sup_type,
    X_lab,
    y_lab,
    X_unl,
    n_features,
    rep,
    unlabeled_rate,
    split_seed,
    label_seed,
    k,
):
    """
    给没有 ranking 的算法使用。
    每个 k 单独 fit，但 selected features 会落盘缓存。
    """
    use_cache = getattr(cfg, "USE_CACHE", True)
    force = getattr(cfg, "FORCE_RECOMPUTE_SELECTOR", False)

    path = _selector_cache_path(
        ds_name=ds_name,
        method_name=method_name,
        rep=rep,
        unlabeled_rate=unlabeled_rate,
        split_seed=split_seed,
        label_seed=label_seed,
        n_features=n_features,
        kind="selected",
        k=k,
    )

    if use_cache and path.exists() and not force:
        t0 = time.time()
        selected = np.load(path)
        return selected.astype(int), {
            "select_mode": "selected_cache",
            "select_time": time.time() - t0,
            "selected_cache_file": str(path),
        }

    t0 = time.time()
    selector = _fit_selector(
        selector_cls=selector_cls,
        sup_type=sup_type,
        X_lab=X_lab,
        y_lab=y_lab,
        X_unl=X_unl,
        n_selected_features=k,
    )

    selected = selector.transform(X_lab).astype(int)
    selected = np.asarray(selected, dtype=int).ravel()
    elapsed = time.time() - t0

    if use_cache:
        np.save(path, selected)

    return selected, {
        "select_mode": "selected_fit",
        "select_time": elapsed,
        "selected_cache_file": str(path),
    }


# ═══════════════════════ 单个并行任务 ═══════════════════════

def _run_one_method_task(
    ds_name,
    X,
    y,
    method_name,
    method_cls,
    rep,
    unlabeled_rate,
    split_seed,
    label_seed_base,
):
    """
    一个任务负责：
      一个数据集 + 一个 repeat + 一个 unlabeled_rate + 一个 method

    任务内部完成：
      1. train/test split
      2. labeled/unlabeled split
      3. selector ranking 或 selected feature cache
      4. 所有 feature_ratio
      5. 所有 classifier
      6. 保存 result shard CSV

    返回：
      result shard CSV 路径
    """
    t_task_start = time.time()

    label_seed = label_seed_base + round(unlabeled_rate * 1000)

    shard_path = _result_shard_path(
        ds_name=ds_name,
        method_name=method_name,
        rep=rep,
        unlabeled_rate=unlabeled_rate,
        split_seed=split_seed,
        label_seed=label_seed,
    )

    force_results = getattr(cfg, "FORCE_RECOMPUTE_RESULTS", False)

    if shard_path.exists() and not force_results:
        print(
            f"[SKIP] existing shard | dataset={ds_name}, rep={rep}, "
            f"ur={unlabeled_rate}, method={method_name}",
            flush=True,
        )
        return str(shard_path)

    try:
        t0 = time.time()
        X_tr, X_te, y_tr, y_te = split_train_test(
            X, y, cfg.TEST_SIZE, split_seed
        )
        split_train_test_time = time.time() - t0

        t0 = time.time()
        X_lab, X_unl, y_lab, _ = split_labeled_unlabeled(
            X_tr, y_tr, unlabeled_rate, label_seed
        )
        split_labeled_time = time.time() - t0

        n_features = X.shape[1]
        sup_type = _get_supervision_type(method_name)

        ranking, ranking_info = _get_or_compute_ranking(
            ds_name=ds_name,
            method_name=method_name,
            selector_cls=method_cls,
            sup_type=sup_type,
            X_lab=X_lab,
            y_lab=y_lab,
            X_unl=X_unl,
            n_features=n_features,
            rep=rep,
            unlabeled_rate=unlabeled_rate,
            split_seed=split_seed,
            label_seed=label_seed,
        )

        classifiers = build_classifiers(cfg.CLASSIFIER_PARAMS)

        rows = []

        print(
            f"[START] dataset={ds_name}, rep={rep}, ur={unlabeled_rate}, "
            f"method={method_name}, selector_mode={ranking_info['selector_mode']}, "
            f"selector_time={ranking_info['selector_time']:.2f}s",
            flush=True,
        )

        for feature_ratio in cfg.FEATURE_RATIOS:
            k = max(1, int(n_features * feature_ratio))

            if ranking is not None:
                t0 = time.time()
                selected_idx = ranking[:k]
                select_info = {
                    "select_mode": "from_ranking",
                    "select_time": time.time() - t0,
                    "selected_cache_file": ranking_info.get("selector_cache_file", ""),
                }
            else:
                selected_idx, select_info = _get_or_compute_selected(
                    ds_name=ds_name,
                    method_name=method_name,
                    selector_cls=method_cls,
                    sup_type=sup_type,
                    X_lab=X_lab,
                    y_lab=y_lab,
                    X_unl=X_unl,
                    n_features=n_features,
                    rep=rep,
                    unlabeled_rate=unlabeled_rate,
                    split_seed=split_seed,
                    label_seed=label_seed,
                    k=k,
                )

            selected_idx = np.asarray(selected_idx, dtype=int).ravel()

            if len(selected_idx) == 0:
                continue

            t0 = time.time()
            X_tr_s = X_tr[:, selected_idx]
            X_te_s = X_te[:, selected_idx]
            slice_time = time.time() - t0

            for clf_name, base_clf in classifiers.items():
                clf = clone(base_clf)

                t0 = time.time()
                acc = evaluate(clf, X_tr_s, y_tr, X_te_s, y_te)
                clf_time = time.time() - t0

                row = {
                    "dataset": ds_name,
                    "method": method_name,
                    "classifier": clf_name,
                    "unlabeled_rate": unlabeled_rate,
                    "feature_ratio": feature_ratio,
                    "repeat": rep,
                    "accuracy": acc,

                    # 计时信息
                    "task_total_time_so_far": time.time() - t_task_start,
                    "split_train_test_time": split_train_test_time,
                    "split_labeled_unlabeled_time": split_labeled_time,
                    "selector_mode": ranking_info["selector_mode"],
                    "selector_time": ranking_info["selector_time"],
                    "select_mode": select_info["select_mode"],
                    "select_time": select_info["select_time"],
                    "slice_time": slice_time,
                    "classifier_time": clf_time,

                    # 诊断信息
                    "n_samples": int(X.shape[0]),
                    "n_features": int(n_features),
                    "n_train": int(X_tr.shape[0]),
                    "n_test": int(X_te.shape[0]),
                    "n_labeled": int(X_lab.shape[0]),
                    "n_unlabeled": int(X_unl.shape[0]),
                    "k_selected": int(len(selected_idx)),
                    "split_seed": int(split_seed),
                    "label_seed": int(label_seed),
                }

                rows.append(row)

                print(
                    f"[DONE] ds={ds_name} | rep={rep} | ur={unlabeled_rate} | "
                    f"method={method_name} | fr={feature_ratio} | clf={clf_name} | "
                    f"acc={acc:.4f} | clf_time={clf_time:.2f}s | "
                    f"select={select_info['select_mode']} | "
                    f"task_time={time.time() - t_task_start:.1f}s",
                    flush=True,
                )

        df = pd.DataFrame(rows)
        _save_csv_atomic(df, shard_path)

        print(
            f"[SHARD SAVED] {shard_path} | rows={len(rows)} | "
            f"time={time.time() - t_task_start:.1f}s",
            flush=True,
        )

        return str(shard_path)

    except Exception as e:
        error_path = shard_path.with_suffix(".error.txt")
        error_text = (
            f"dataset={ds_name}\n"
            f"method={method_name}\n"
            f"repeat={rep}\n"
            f"unlabeled_rate={unlabeled_rate}\n"
            f"split_seed={split_seed}\n"
            f"label_seed={label_seed}\n\n"
            f"{repr(e)}\n\n"
            f"{traceback.format_exc()}"
        )
        error_path.write_text(error_text, encoding="utf-8")

        print(
            f"[ERROR] dataset={ds_name}, rep={rep}, ur={unlabeled_rate}, "
            f"method={method_name}. See: {error_path}",
            flush=True,
        )

        raise


# ═══════════════════════ 合并分片结果 ═══════════════════════

def _merge_result_shards(shard_paths):
    shard_paths = [Path(p) for p in shard_paths]
    shard_paths = [p for p in shard_paths if p.exists() and p.stat().st_size > 0]

    if len(shard_paths) == 0:
        raise RuntimeError("No result shards found. Please check whether tasks failed.")

    frames = []
    for p in shard_paths:
        try:
            df = pd.read_csv(p)
            if len(df) > 0:
                frames.append(df)
        except Exception as e:
            print(f"[WARN] failed to read shard: {p}, error={e}", flush=True)

    if len(frames) == 0:
        raise RuntimeError("All result shards are empty or unreadable.")

    return pd.concat(frames, ignore_index=True)


def _save_timing_report(df: pd.DataFrame):
    timing_dir = cfg.OUTPUT_DIR / "timing"
    timing_dir.mkdir(parents=True, exist_ok=True)

    # 1. selector 耗时：按 dataset/method/repeat/unlabeled_rate 去重
    selector_cols = [
        "dataset",
        "method",
        "repeat",
        "unlabeled_rate",
        "selector_mode",
        "selector_time",
        "n_samples",
        "n_features",
        "n_labeled",
        "n_unlabeled",
    ]

    selector_df = (
        df[selector_cols]
        .drop_duplicates()
        .sort_values(["selector_time"], ascending=False)
    )
    selector_path = timing_dir / "selector_timing.csv"
    selector_df.to_csv(selector_path, index=False, float_format="%.6f")

    # 2. classifier 耗时
    classifier_cols = [
        "dataset",
        "method",
        "classifier",
        "feature_ratio",
        "unlabeled_rate",
        "repeat",
        "classifier_time",
        "k_selected",
        "n_train",
        "n_test",
    ]
    classifier_df = df[classifier_cols].sort_values(
        ["classifier_time"], ascending=False
    )
    classifier_path = timing_dir / "classifier_timing.csv"
    classifier_df.to_csv(classifier_path, index=False, float_format="%.6f")

    # 3. 汇总平均耗时
    summary = (
        df.groupby(["dataset", "method", "classifier"], as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_classifier_time=("classifier_time", "mean"),
            max_classifier_time=("classifier_time", "max"),
            mean_selector_time=("selector_time", "mean"),
            max_selector_time=("selector_time", "max"),
            n_rows=("accuracy", "size"),
        )
        .sort_values(["mean_selector_time", "mean_classifier_time"], ascending=False)
    )
    summary_path = timing_dir / "timing_summary.csv"
    summary.to_csv(summary_path, index=False, float_format="%.6f")

    print(f"[TIMING] selector timing:   {selector_path}")
    print(f"[TIMING] classifier timing: {classifier_path}")
    print(f"[TIMING] timing summary:    {summary_path}")


def _save_results_with_manager(df: pd.DataFrame):
    """
    复用你原来的 ResultsManager。
    原 main.py 使用 results.add(...) 后保存 summary/detailed/raw，再画图。
    这里把合并后的 df 重新灌入 ResultsManager。
    """
    results = ResultsManager()

    required_cols = [
        "dataset",
        "method",
        "classifier",
        "unlabeled_rate",
        "feature_ratio",
        "repeat",
        "accuracy",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Merged dataframe missing columns: {missing}")

    for row in df[required_cols].itertuples(index=False):
        results.add(
            dataset=row.dataset,
            method=row.method,
            classifier=row.classifier,
            unlabeled_rate=row.unlabeled_rate,
            feature_ratio=row.feature_ratio,
            repeat=int(row.repeat),
            accuracy=float(row.accuracy),
        )

    print("\n--- Saving CSV ---")
    results.save_summary_csv(cfg.OUTPUT_DIR)
    results.save_detailed_csv(cfg.OUTPUT_DIR)

    raw = cfg.OUTPUT_DIR / "raw_results.csv"
    df.to_csv(raw, index=False, float_format="%.6f")
    print(f"[CSV] {raw}")

    print("\n--- Generating Figures ---")
    results.plot_acc_vs_unlabeled_rate(
        cfg.FIGURE_DIR,
        cfg.FIG_FORMAT,
        cfg.FIG_DPI,
    )
    results.plot_acc_vs_feature_ratio(
        cfg.FIGURE_DIR,
        cfg.FIG_FORMAT,
        cfg.FIG_DPI,
    )

    return results


# ═══════════════════════ 主流程 ═══════════════════════

def run_all_parallel():
    _ensure_dirs()

    seeds = generate_seeds(cfg.BASE_SEED, cfg.N_REPEATS * 2)
    split_seeds = seeds[: cfg.N_REPEATS]
    label_seeds = seeds[cfg.N_REPEATS :]

    all_shard_paths = []
    t_all = time.time()

    for ds_cfg in cfg.DATASETS:
        ds_name = ds_cfg["name"]

        print(f"\n{'=' * 80}")
        print(f"Dataset: {ds_name}")
        print(f"{'=' * 80}")

        t0 = time.time()
        X, y = load_data(ds_cfg, cfg.DATA_DIR)
        X, y = preprocess(X, y)
        load_time = time.time() - t0

        print(
            f"[DATA] {ds_name}: X.shape={X.shape}, y.shape={y.shape}, "
            f"load+preprocess={load_time:.2f}s",
            flush=True,
        )

        tasks = []
        for rep in range(cfg.N_REPEATS):
            split_seed = split_seeds[rep]
            label_seed_base = label_seeds[rep]

            for unlabeled_rate in cfg.UNLABELED_RATES:
                for method_name, method_cls in METHODS.items():
                    tasks.append(
                        (
                            ds_name,
                            X,
                            y,
                            method_name,
                            method_cls,
                            rep,
                            unlabeled_rate,
                            split_seed,
                            label_seed_base,
                        )
                    )

        print(
            f"[TASKS] dataset={ds_name}, tasks={len(tasks)}, "
            f"n_jobs={getattr(cfg, 'PARALLEL_N_JOBS', 1)}",
            flush=True,
        )

        n_jobs = getattr(cfg, "PARALLEL_N_JOBS", 1)
        max_nbytes = getattr(cfg, "JOBLIB_MAX_NBYTES", "64M")

        shard_paths = Parallel(
            n_jobs=n_jobs,
            backend="loky",
            verbose=10,
            max_nbytes=max_nbytes,
            mmap_mode="r",
        )(
            delayed(_run_one_method_task)(*task)
            for task in tasks
        )

        all_shard_paths.extend(shard_paths)

        print(
            f"[DATASET DONE] {ds_name} | time={time.time() - t0:.1f}s | "
            f"shards={len(shard_paths)}",
            flush=True,
        )

    print(f"\n[MERGE] merging {len(all_shard_paths)} shards...")
    merged_df = _merge_result_shards(all_shard_paths)

    _save_timing_report(merged_df)
    results = _save_results_with_manager(merged_df)

    print(f"\n✓ All experiments completed in {time.time() - t_all:.1f}s.")

    return results


def main():
    print("=" * 80)
    print("  Semi-Supervised Feature Selection Experiment - Parallel Version")
    print("=" * 80)

    run_all_parallel()


if __name__ == "__main__":
    main()