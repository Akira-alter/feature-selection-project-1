"""
结果管理器：
- 以行记录收集所有原始结果
- 按 (dataset, classifier) 汇总输出 CSV
- 自动保存命名规范的图表
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")                     # 非交互后端，不弹窗
import matplotlib.pyplot as plt
from pathlib import Path


class ResultsManager:
    """
    每条记录 7 个维度:
        dataset / method / classifier / unlabeled_rate / feature_ratio / repeat / accuracy
    """

    def __init__(self):
        self._records: list[dict] = []

    def add(self, *, dataset, method, classifier,
            unlabeled_rate, feature_ratio, repeat, accuracy):
        self._records.append({
            "dataset":        dataset,
            "method":         method,
            "classifier":     classifier,
            "unlabeled_rate": unlabeled_rate,
            "feature_ratio":  feature_ratio,
            "repeat":         repeat,
            "accuracy":       accuracy,
        })

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._records)

    # ───────────── CSV 输出 ─────────────

    def save_summary_csv(self, output_dir: Path):
        """
        同一数据集 + 同一分类器 → 一个 CSV
        行 = unlabeled_rate，列 = 各算法，值 = 平均准确率
        (跨 repeat 和 feature_ratio 取平均)
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()

        for (ds, clf), grp in df.groupby(["dataset", "classifier"]):
            pivot = grp.pivot_table(
                index="unlabeled_rate", columns="method",
                values="accuracy", aggfunc="mean"
            )
            pivot.index.name = "Unlabeled Rate"
            path = output_dir / f"{ds}_{clf}_summary.csv"
            pivot.to_csv(path, float_format="%.4f")
            print(f"  [CSV] {path}")

    #有其它两种保存方法
    # def save_summary_csv(self, output_dir: Path):
    #     output_dir.mkdir(parents=True, exist_ok=True)
    #     df = self.to_dataframe()
    #
    #     for (ds, clf), grp in df.groupby(["dataset", "classifier"]):
    #         # 方案 A：每个 feature_ratio 一张表
    #         for fr, fr_grp in grp.groupby("feature_ratio"):
    #             pivot = fr_grp.pivot_table(
    #                 index="unlabeled_rate", columns="method",
    #                 values="accuracy", aggfunc="mean"
    #             )
    #             pivot.index.name = "Unlabeled Rate"
    #             path = output_dir / f"{ds}_{clf}_fr{int(fr * 100)}.csv"
    #             pivot.to_csv(path, float_format="%.4f")
    #
    #         # 方案 B（可选）：跨 feature_ratio 取最佳
    #         best = grp.loc[
    #             grp.groupby(["method", "unlabeled_rate", "repeat"])["accuracy"]
    #             .idxmax()
    #         ]
    #         pivot_best = best.pivot_table(
    #             index="unlabeled_rate", columns="method",
    #             values="accuracy", aggfunc="mean"
    #         )
    #         path = output_dir / f"{ds}_{clf}_best_ratio.csv"
    #         pivot_best.to_csv(path, float_format="%.4f")

    def save_detailed_csv(self, output_dir: Path):
        """
        保留 (unlabeled_rate, feature_ratio) 两个维度，仅跨 repeat 取均值。
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()

        for (ds, clf), grp in df.groupby(["dataset", "classifier"]):
            pivot = grp.pivot_table(
                index=["unlabeled_rate", "feature_ratio"],
                columns="method", values="accuracy", aggfunc="mean"
            )
            path = output_dir / f"{ds}_{clf}_detailed.csv"
            pivot.to_csv(path, float_format="%.4f")
            print(f"  [CSV] {path}")

    # ───────────── 绘图 ─────────────

    def plot_acc_vs_unlabeled_rate(self, fig_dir: Path,
                                   fmt="png", dpi=300):
        """每个 (dataset, classifier) 一张：准确率 vs 未标记率。"""
        fig_dir.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()

        for (ds, clf), grp in df.groupby(["dataset", "classifier"]):
            fig, ax = plt.subplots(figsize=(10, 6))
            agg = (grp.groupby(["method", "unlabeled_rate"])["accuracy"]
                       .mean().reset_index())

            for method, mg in agg.groupby("method"):
                mg = mg.sort_values("unlabeled_rate")
                ax.plot(mg["unlabeled_rate"], mg["accuracy"],
                        marker="o", label=method)

            ax.set_title(f"{ds} — {clf.upper()}: Accuracy vs Unlabeled Rate")
            ax.set_xlabel("Unlabeled Rate")
            ax.set_ylabel("Average Accuracy")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=8)
            fig.tight_layout()

            path = fig_dir / f"{ds}_{clf}_vs_unlabeled.{fmt}"
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            print(f"  [FIG] {path}")

    def plot_acc_vs_feature_ratio(self, fig_dir: Path,
                                   fmt="png", dpi=300):
        """每个 (dataset, classifier, unlabeled_rate) 一张：准确率 vs 特征比例。"""
        fig_dir.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()

        for (ds, clf, ur), grp in df.groupby(
                ["dataset", "classifier", "unlabeled_rate"]):
            fig, ax = plt.subplots(figsize=(10, 6))
            agg = (grp.groupby(["method", "feature_ratio"])["accuracy"]
                       .mean().reset_index())

            for method, mg in agg.groupby("method"):
                mg = mg.sort_values("feature_ratio")
                ax.plot(mg["feature_ratio"], mg["accuracy"],
                        marker="o", label=method)

            ax.set_title(f"{ds} — {clf.upper()} (unlabeled={ur:.0%})")
            ax.set_xlabel("Feature Ratio")
            ax.set_ylabel("Average Accuracy")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=8)
            fig.tight_layout()

            path = fig_dir / f"{ds}_{clf}_ur{int(ur*100)}_vs_features.{fmt}"
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            print(f"  [FIG] {path}")