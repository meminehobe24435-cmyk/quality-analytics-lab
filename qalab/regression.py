"""回归分析：预测**连续**的性能指标，而不是二值的合格/不合格。

**用途**：破坏性性能测试要抽检、要报废样品。如果能用「检验之前就知道的工艺参数」
把力学性能（拉伸强度等）预测出来，就能：
  1. 只在预测值接近规格边界时才做破坏性抽检（省成本）；
  2. 提前发现工艺往规格边缘漂移的趋势。

做法：
  - 目标：连续的性能指标（默认拉伸强度 tensile_strength_mpa），
    只用在性能测试表里**真实测到**的单元（目标缺失的行直接丢弃，不做填补 ——
    填补目标等于伪造标签）；
  - 特征：与分类模型同一套「检验前可得」的特征（工艺参数 + 原料参数），
    同样剔除检验环节字段，避免泄漏；
  - 模型：线性回归（可解释基准）+ 梯度提升回归（非线性对照）；
  - 切分：固定种子的随机切分（回归不需要分层，但同样固定种子）；
  - 指标：MAE / RMSE / R²，并**必须**与「直接用均值预测」的基线对比 ——
    没有超过均值基线的回归模型等于没用。

退化输入（全部有测试）：目标列方差为 0、目标全缺失、样本量过少。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from . import config as cfg
from .model import build_preprocessor, prepare_features

__all__ = [
    "regression_metrics",
    "build_regressors",
    "run_regression",
    "DEFAULT_TARGETS",
]

DEFAULT_TARGETS: tuple[str, ...] = cfg.PERF_NUMERIC
"""可回归的连续性能指标（破坏性抽检测到的量）。"""


def regression_metrics(y_true, y_pred) -> dict[str, object]:
    """MAE / RMSE / R²，并显式处理退化输入。

    退化情形：
      - 目标方差为 0（全部取值相同）：R² 在数学上无定义（分母为 0），
        返回 None 而不是 sklearn 那个"看起来正常"的 0.0 或 1.0；
      - 样本为空：全部返回 None。
    """
    y_true = pd.Series(y_true, dtype="float64").to_numpy()
    y_pred = pd.Series(y_pred, dtype="float64").to_numpy()
    out: dict[str, object] = {
        "n": int(len(y_true)),
        "mae": None,
        "rmse": None,
        "r2": None,
        "target_mean": None,
        "target_std": None,
        "constant_target": False,
        "note": "",
    }
    if len(y_true) == 0:
        out["note"] = "没有可用样本"
        return out
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) == 0:
        out["n"] = 0
        out["note"] = "预测值与真值没有重叠的有效样本"
        return out
    out["n"] = int(len(y_true))
    out["mae"] = float(mean_absolute_error(y_true, y_pred))
    out["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    out["target_mean"] = float(np.mean(y_true))
    out["target_std"] = float(np.std(y_true, ddof=0))
    if out["target_std"] == 0:
        out["constant_target"] = True
        out["note"] = "目标方差为 0，R² 无定义（分母为 0），返回 None"
    else:
        out["r2"] = float(r2_score(y_true, y_pred))
    return out


def build_regressors(seed: int = cfg.SEED) -> dict[str, object]:
    """两个回归模型：线性回归（可解释基准）与梯度提升回归（非线性对照）。"""
    return {
        "linear_regression": LinearRegression(),
        "gradient_boosting_regressor": HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.08,
            random_state=seed,
            early_stopping=False,
        ),
    }


def _fit_one_target(
    df: pd.DataFrame,
    target: str,
    seed: int,
    test_size: float = 0.20,
    detailed: bool = False,
) -> tuple[dict[str, object], dict[str, object]]:
    """对单个目标列跑回归，返回 (结果, 绘图数据)。"""
    result: dict[str, object] = {
        "target": target,
        "n_rows_total": int(len(df)),
        "n_rows_used": 0,
        "split": {},
        "models": {},
        "baseline_mean": {},
        "improvement_over_baseline": {},
        "note": "",
    }
    extras: dict[str, object] = {}

    if target not in df.columns:
        result["note"] = f"缺少目标列 {target}"
        return result, extras

    y_all = pd.to_numeric(df[target], errors="coerce")
    usable = y_all.notna()
    result["n_rows_used"] = int(usable.sum())
    result["missing_target_rows"] = int((~usable).sum())
    if usable.sum() < 30:
        result["note"] = (
            f"有 {target} 有效值的样本只有 {int(usable.sum())} 条，样本量不足以做回归"
        )
        return result, extras
    if float(y_all[usable].std(ddof=0)) == 0:
        result["note"] = f"目标 {target} 的方差为 0（所有取值相同），回归无意义"
        result["constant_target"] = True
        return result, extras

    sub = df.loc[usable]
    X, numeric, categorical = prepare_features(sub)
    y = pd.to_numeric(sub[target], errors="coerce")

    idx = np.arange(len(X))
    idx_train, idx_test = train_test_split(idx, test_size=test_size, random_state=seed)
    X_train, X_test = X.iloc[idx_train], X.iloc[idx_test]
    y_train, y_test = y.iloc[idx_train], y.iloc[idx_test]
    result["split"] = {"train": int(len(idx_train)), "test": int(len(idx_test))}
    result["feature_columns"] = list(X.columns)
    result["excluded_leaky_columns"] = [c for c in cfg.LEAKY_COLUMNS if c in sub.columns]

    # 基线：直接用训练集均值预测（回归的"什么都没学"下限）
    baseline = Pipeline(
        [("prep", build_preprocessor(numeric, categorical)), ("reg", DummyRegressor(strategy="mean"))]
    )
    baseline.fit(X_train, y_train)
    base_test = regression_metrics(y_test, baseline.predict(X_test))
    base_train = regression_metrics(y_train, baseline.predict(X_train))
    result["baseline_mean"] = {"train": base_train, "test": base_test}

    fitted: dict[str, Pipeline] = {}
    for name, estimator in build_regressors(seed).items():
        model = Pipeline([("prep", build_preprocessor(numeric, categorical)), ("reg", estimator)])
        model.fit(X_train, y_train)
        fitted[name] = model
        test_pred = model.predict(X_test)
        train_pred = model.predict(X_train)
        entry = {
            "train": regression_metrics(y_train, train_pred),
            "test": regression_metrics(y_test, test_pred),
        }
        result["models"][name] = entry
        if base_test["mae"] and base_test["mae"] > 0:
            entry["mae_reduction_vs_baseline_pct"] = round(
                (base_test["mae"] - entry["test"]["mae"]) / base_test["mae"] * 100, 4
            )
        if detailed:
            extras.setdefault("pred_test", {})[name] = test_pred
            extras.setdefault("pred_train", {})[name] = train_pred
            extras.setdefault("residuals", {})[name] = y_test.to_numpy() - test_pred

    if detailed and fitted:
        extras["target"] = target
        extras["y_test"] = y_test.to_numpy()
        extras["y_train"] = y_train.to_numpy()
        extras["baseline_test"] = baseline.predict(X_test)
        extras["best_model"] = min(
            result["models"], key=lambda n: result["models"][n]["test"]["mae"]
        )
    return result, extras


def run_regression(
    df: pd.DataFrame,
    target: str = cfg.PERF_NUMERIC[0],
    seed: int = cfg.SEED,
    all_targets: tuple[str, ...] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """跑回归分析：主目标给全部细节，其余目标给一张对比表。

    返回 (可 JSON 化的结果, 绘图数据)。第二个返回值只在主目标上生成一条
    「预测值 vs 真值」的数据。
    """
    detailed, extras = _fit_one_target(df, target, seed, detailed=True)
    result: dict[str, object] = {
        "purpose": "预测连续的性能指标，用于减少破坏性抽检、提前发现往规格边缘的漂移",
        "primary_target": target,
        "primary": detailed,
        "all_targets": [],
        "note": "",
    }

    for other in all_targets or DEFAULT_TARGETS:
        if other == target:
            continue
        entry, _ = _fit_one_target(df, other, seed, detailed=False)
        best_name = None
        best_mae = None
        for name, m in entry.get("models", {}).items():
            mae = m["test"]["mae"]
            if mae is not None and (best_mae is None or mae < best_mae):
                best_mae, best_name = mae, name
        result["all_targets"].append(
            {
                "target": other,
                "n_rows_used": entry["n_rows_used"],
                "baseline_mae": entry.get("baseline_mean", {}).get("test", {}).get("mae"),
                "best_model": best_name,
                "best_mae": best_mae,
                "best_rmse": entry["models"][best_name]["test"]["rmse"] if best_name else None,
                "best_r2": entry["models"][best_name]["test"]["r2"] if best_name else None,
                "note": entry["note"],
            }
        )

    if detailed.get("models"):
        best = min(
            detailed["models"], key=lambda n: detailed["models"][n]["test"]["mae"]
        )
        result["best_model_by_test_mae"] = best
        base_mae = detailed["baseline_mean"]["test"]["mae"]
        best_mae = detailed["models"][best]["test"]["mae"]
        result["baseline_mae"] = base_mae
        result["best_mae"] = best_mae
        result["beats_baseline"] = bool(base_mae is not None and best_mae is not None and best_mae < base_mae)
        result["note"] = (
            f"主目标 {target}：最优模型 {best} 的测试集 MAE={best_mae:.4f}，"
            f"均值基线 MAE={base_mae:.4f}，"
            f"降低 {(base_mae - best_mae) / base_mae * 100:.2f}%"
        )
    else:
        result["note"] = detailed.get("note", "回归未能完成")
    return result, extras
