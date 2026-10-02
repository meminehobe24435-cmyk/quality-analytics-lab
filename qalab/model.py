"""预测性模型：预测「会不会不合格」。

设计要点（都是为了避免最常见的建模错误）：
  1. **信息泄漏控制**：特征只用「检验之前就知道」的工艺参数与原料参数。
     检验方法/检验员/缺陷类型是检验环节才产生的，与标签同期甚至因果倒置，
     纳入特征会让指标虚高到 AP≈0.99（开发中实测过，见 README 的踩坑记录）。
     性能测试是**破坏性抽检**，同样在判定之后，因此主模型不含它们；
     带性能测试特征的版本只作为对照（ablation），并在报告里标注不可部署。
  2. **分层切分**：train 64% / validation 16% / test 20%，全部 stratify，
     保证三个集合的不合格率一致。
  3. **阈值只在验证集上选**：测试集只用选好的阈值评估**一次**。
  4. **类别不平衡**：class_weight='balanced'；评估主指标用 AP（PR-AUC）与
     ROC-AUC，而不是只看 accuracy（不合格率只有约 12%，全预测为合格的
     accuracy 就有 88%）。
  5. **分层交叉验证**：StratifiedKFold，报告各模型 CV 均值与标准差。
  6. **可复现**：全部 random_state 固定，n_jobs=1（多线程会引入不确定性）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as cfg

__all__ = [
    "build_preprocessor",
    "build_models",
    "split_data",
    "compute_metrics",
    "select_threshold",
    "cross_validate_models",
    "run_modeling",
    "extract_feature_importance",
    "prepare_features",
]

SCORERS: dict[str, str] = {
    "average_precision": "average_precision",
    "roc_auc": "roc_auc",
    "f1": "f1",
    "precision": "precision",
    "recall": "recall",
}


def prepare_features(
    df: pd.DataFrame, include_perf: bool = False
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """挑出建模特征，返回 (X, 数值列, 类别列)。

    `include_perf=True` 时额外加入破坏性性能测试指标（含缺失指示列），
    仅用于 ablation 对照：这些量在检验判定之后才测得，不能用于上线预测。
    """
    numeric = [c for c in cfg.MODEL_NUMERIC if c in df.columns]
    categorical = [c for c in cfg.MODEL_CATEGORICAL if c in df.columns]
    if include_perf:
        for col in cfg.PERF_NUMERIC:
            flag = f"{col}_was_missing"
            if flag in df.columns:
                # 缺失本身可能携带信息（未测 = 被判定为高风险而不做破坏性测试）
                df = df.assign(**{flag: df[flag].astype(int)})
                numeric.append(flag)
            if col in df.columns:
                numeric.append(col)
    X = df[numeric + categorical].copy()
    return X, numeric, categorical


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """数值列：中位数填补 + 标准化；类别列：常数填补 + one-hot。

    填补与标准化都放在 Pipeline 内部，因此交叉验证时每个 fold 单独拟合，
    不会出现「用全量数据算出来的中位数」泄漏进验证折的问题。
    """
    numeric_pipe = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("num", numeric_pipe, numeric), ("cat", categorical_pipe, categorical)],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_models(seed: int = cfg.SEED) -> dict[str, Pipeline]:
    """三个对比模型（此时只含分类器，预处理由 `build_pipeline` 拼上）。

    逻辑回归：线性基准，系数可解释，便于向工艺/质量同事解释方向与量级；
    随机森林：非线性 + 交互，对异常值稳健；
    梯度提升（HistGB）：表格数据上通常精度最高，但可解释性最弱。

    三者都显式处理类别不平衡（class_weight），且 n_jobs=1 保证可复现。
    """
    return {
        "logistic_regression": Pipeline(
            [
                (
                    "clf",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=seed,
                    ),
                )
            ]
        ),
        "random_forest": Pipeline(
            [
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        random_state=seed,
                        n_jobs=1,
                    ),
                )
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        max_iter=200,
                        learning_rate=0.08,
                        class_weight="balanced",
                        random_state=seed,
                        early_stopping=False,
                    ),
                )
            ]
        ),
    }


def build_pipeline(
    name: str,
    numeric: list[str],
    categorical: list[str],
    seed: int = cfg.SEED,
) -> Pipeline:
    """拼出完整可训练管线：预处理 + 分类器。"""
    models = build_models(seed)
    if name not in models:
        raise KeyError(f"未知模型: {name}；可选 {sorted(models)}")
    return Pipeline([("prep", build_preprocessor(numeric, categorical))] + list(models[name].steps))



def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int = cfg.SEED,
    test_size: float = cfg.TEST_SIZE,
    val_size: float = cfg.VAL_SIZE,
) -> dict[str, object]:
    """分层切成 train / validation / test，并返回索引以便断言互不重叠。

    先切出 test（占总样本 test_size），再从剩余里切出 validation。
    """
    idx = np.arange(len(X))
    idx_train_full, idx_test = train_test_split(
        idx, test_size=test_size, stratify=y.to_numpy(), random_state=seed
    )
    idx_train, idx_val = train_test_split(
        idx_train_full,
        test_size=val_size,
        stratify=y.to_numpy()[idx_train_full],
        random_state=seed,
    )
    return {
        "X_train": X.iloc[idx_train],
        "y_train": y.iloc[idx_train],
        "X_val": X.iloc[idx_val],
        "y_val": y.iloc[idx_val],
        "X_test": X.iloc[idx_test],
        "y_test": y.iloc[idx_test],
        "idx_train": idx_train,
        "idx_val": idx_val,
        "idx_test": idx_test,
    }


def compute_metrics(y_true, y_pred, y_proba=None) -> dict[str, object]:
    """计算分类指标，并显式处理**退化输入**。

    退化情形与处理方式（均有单元测试覆盖）：
      - 标签只有单一类别（全 0 或全 1）：ROC-AUC 与 AP 无定义 -> 返回 None，
        而不是给出一个看起来"完美"的 1.0（sklearn 对全 1 的 AP 会返回 1.0，
        具有误导性）。
      - 没有任何正例预测：precision 无定义 -> 按 0.0 处理（zero_division=0），
        并在 `degenerate` 里标注，避免把 NaN 混进报告。
    """
    y_true = pd.Series(y_true).to_numpy()
    y_pred = pd.Series(y_pred).to_numpy()
    out: dict[str, object] = {
        "n": int(len(y_true)),
        "n_positive": int((y_true == 1).sum()),
        "n_negative": int((y_true == 0).sum()),
        "positive_rate": float((y_true == 1).mean()) if len(y_true) else None,
        "single_class": bool(len(np.unique(y_true)) < 2),
        "no_positive_prediction": bool((y_pred == 1).sum() == 0),
        "average_precision": None,
        "roc_auc": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "accuracy": None,
        "balanced_accuracy": None,
        "specificity": None,
        "brier_score": None,
        "confusion_matrix": None,
    }
    if len(y_true) == 0:
        return out

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    out["confusion_matrix"] = [[tn, fp], [fn, tp]]
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else None
    if len(np.unique(y_true)) > 1:
        out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    if y_proba is not None and len(np.unique(y_true)) > 1:
        proba = np.asarray(y_proba, dtype=float)
        out["average_precision"] = float(average_precision_score(y_true, proba))
        out["roc_auc"] = float(roc_auc_score(y_true, proba))
        out["brier_score"] = float(brier_score_loss(y_true, proba))
    return out


def select_threshold(y_true, y_proba, metric: str = "f1") -> dict[str, object]:
    """在验证集上按目标指标选阈值（默认最大化 F1）。

    候选阈值取所有出现过的概率值，因此结果是离散且确定的；
    argmax 取第一个最大值，保证同分时结果可复现。
    退化情形（标签单一类别、无候选阈值）返回默认阈值 0.5 并标注。
    """
    y_true = pd.Series(y_true).to_numpy()
    proba = np.asarray(y_proba, dtype=float)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2 or len(proba) == 0:
        return {
            "threshold": 0.5,
            "metric": metric,
            "score": None,
            "n_candidates": 0,
            "note": "验证集标签只有单一类别或为空，回退到默认阈值 0.5",
        }
    candidates = np.unique(proba)
    scores = np.array(
        [
            f1_score(y_true, (proba >= t).astype(int), zero_division=0)
            if metric == "f1"
            else precision_score(y_true, (proba >= t).astype(int), zero_division=0)
            for t in candidates
        ]
    )
    best = int(np.argmax(scores))
    return {
        "threshold": float(candidates[best]),
        "metric": metric,
        "score": float(scores[best]),
        "n_candidates": int(len(candidates)),
        "note": "在验证集上选出",
    }


def cross_validate_models(
    X: pd.DataFrame,
    y: pd.Series,
    numeric: list[str],
    categorical: list[str],
    seed: int = cfg.SEED,
    folds: int = cfg.CV_FOLDS,
) -> dict[str, dict[str, float]]:
    """分层 K 折交叉验证，返回 {模型名: {指标: {"mean":…, "std":…}}}。"""
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    results: dict[str, dict[str, float]] = {}
    for name in build_models(seed):
        model = build_pipeline(name, numeric, categorical, seed)
        scores = cross_validate(
            model, X, y, cv=cv, scoring=SCORERS, n_jobs=1, error_score="raise"
        )
        results[name] = {}
        for metric in SCORERS:
            values = scores[f"test_{metric}"]
            results[name][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=0)),
                "folds": [float(v) for v in values],
            }
    return results


def extract_feature_importance(
    model: Pipeline,
    top_k: int = 15,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    seed: int = cfg.SEED,
) -> list[dict[str, object]]:
    """提取特征重要性。

    树模型：`feature_importances_`（不纯度的平均下降）；
    线性模型：|标准化后的系数|，并保留正负号作为风险方向；
    两者都没有的模型（如 HistGradientBoostingClassifier 在 sklearn 里不暴露
    原生重要性）：回退到**置换重要性**（在验证集上打乱单列、看 AP 掉多少），
    需要传入 X_val / y_val，否则返回空列表。
    """
    prep: ColumnTransformer = model.named_steps["prep"]
    clf = model.named_steps["clf"]
    try:
        names = list(prep.get_feature_names_out())
    except Exception:  # pragma: no cover - 仅在未拟合时发生
        return []
    clean_names = [n.split("__", 1)[1] if "__" in n else n for n in names]

    if hasattr(clf, "feature_importances_"):
        values = np.asarray(clf.feature_importances_, dtype=float)
        direction = None
        source = "native_importance"
    elif hasattr(clf, "coef_"):
        coef = np.asarray(clf.coef_, dtype=float).ravel()
        values = np.abs(coef)
        direction = coef
        source = "standardized_coefficient"
    elif X_val is not None and y_val is not None and len(X_val):
        from sklearn.inspection import permutation_importance

        pi = permutation_importance(
            model,
            X_val,
            y_val,
            scoring="average_precision",
            n_repeats=5,
            random_state=seed,
            n_jobs=1,
        )
        order = np.argsort(-pi.importances_mean)
        return [
            {
                "feature": str(X_val.columns[i]),
                "importance": float(pi.importances_mean[i]),
                "importance_std": float(pi.importances_std[i]),
                "importance_source": "permutation_importance_on_validation",
            }
            for i in order[:top_k]
        ]
    else:  # pragma: no cover
        return []

    order = np.argsort(-values)
    out: list[dict[str, object]] = []
    for i in order[:top_k]:
        item: dict[str, object] = {
            "feature": clean_names[i] if i < len(clean_names) else f"f{i}",
            "importance": float(values[i]),
            "importance_source": source,
        }
        if direction is not None:
            item["direction"] = "风险升高" if direction[i] > 0 else "风险降低"
            item["coefficient"] = float(direction[i])
        out.append(item)
    return out



def run_modeling(
    df: pd.DataFrame,
    label: str = cfg.LABEL_COL,
    seed: int = cfg.SEED,
    with_perf_ablation: bool = True,
) -> tuple[dict[str, object], dict[str, object]]:
    """跑完整的建模流程。

    返回 (result, extras)：
      result: 可 JSON 化的结果（写进 metrics.json）
      extras: 绘图需要的内存对象（测试集概率、特征重要性明细等），不落盘
    """
    X, numeric, categorical = prepare_features(df, include_perf=False)
    y = pd.to_numeric(df[label], errors="coerce").astype(int)

    splits = split_data(X, y, seed=seed)
    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val, y_val = splits["X_val"], splits["y_val"]
    X_test, y_test = splits["X_test"], splits["y_test"]

    cv_summary = cross_validate_models(X_train, y_train, numeric, categorical, seed=seed)

    result: dict[str, object] = {
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "feature_columns": list(X.columns),
        "numeric_features": numeric,
        "categorical_features": categorical,
        "excluded_leaky_columns": [c for c in cfg.LEAKY_COLUMNS if c in df.columns],
        "label_positive_rate": float(y.mean()),
        "split_sizes": {
            "train": int(len(X_train)),
            "validation": int(len(X_val)),
            "test": int(len(X_test)),
        },
        "split_positive_rates": {
            "train": float(y_train.mean()),
            "validation": float(y_val.mean()),
            "test": float(y_test.mean()),
        },
        "cv_folds": cfg.CV_FOLDS,
        "cv": {k: v for k, v in cv_summary.items()},
        "models": {},
    }

    extras: dict[str, object] = {
        "y_test": y_test.to_numpy(),
        "y_val": y_val.to_numpy(),
        "test_proba": {},
        "val_proba": {},
        "confusion_matrices": {},
        "feature_importance": {},
        "splits": {k: splits[k] for k in ("idx_train", "idx_val", "idx_test")},
        "y_all": y.to_numpy(),
    }

    for name in build_models(seed):
        model = build_pipeline(name, numeric, categorical, seed)
        model.fit(X_train, y_train)
        val_proba = model.predict_proba(X_val)[:, 1]
        threshold = select_threshold(y_val, val_proba)
        # 测试集只在这里被评估一次
        test_proba = model.predict_proba(X_test)[:, 1]
        test_pred = (test_proba >= threshold["threshold"]).astype(int)
        val_pred = (val_proba >= threshold["threshold"]).astype(int)

        result["models"][name] = {
            "threshold_selection": threshold,
            "validation_metrics": compute_metrics(y_val, val_pred, val_proba),
            "test_metrics": compute_metrics(y_test, test_pred, test_proba),
            "top_features": extract_feature_importance(
                model, top_k=12, X_val=X_val, y_val=y_val, seed=seed
            ),
        }
        extras["test_proba"][name] = test_proba
        extras["val_proba"][name] = val_proba
        extras["confusion_matrices"][name] = result["models"][name]["test_metrics"][
            "confusion_matrix"
        ]
        extras["feature_importance"][name] = result["models"][name]["top_features"]

    # 主模型：按验证集 AP 选最优（用验证集选模型，不用测试集）
    best_name = max(
        result["models"],
        key=lambda n: (
            result["models"][n]["validation_metrics"]["average_precision"] or -1.0
        ),
    )
    result["best_model_by_validation_ap"] = best_name
    extras["best_model"] = best_name

    if with_perf_ablation:
        result["ablation_with_perf_features"] = _ablation_perf(
            df, label, seed, best_name
        )
    return result, extras


def _ablation_perf(
    df: pd.DataFrame, label: str, seed: int, model_name: str
) -> dict[str, object]:
    """对照实验：把破坏性性能测试指标也放进特征，观察指标变化。

    这些量在检验判定之后测得（事后信息），因此**不可部署**；
    列出来是为了量化「泄漏能让指标虚高多少」。
    """
    X, numeric, categorical = prepare_features(df, include_perf=True)
    y = pd.to_numeric(df[label], errors="coerce").astype(int)
    splits = split_data(X, y, seed=seed)
    model = build_pipeline(model_name, numeric, categorical, seed)
    model.fit(splits["X_train"], splits["y_train"])
    proba = model.predict_proba(splits["X_test"])[:, 1]
    threshold = select_threshold(
        splits["y_val"], model.predict_proba(splits["X_val"])[:, 1]
    )
    pred = (proba >= threshold["threshold"]).astype(int)
    metrics = compute_metrics(splits["y_test"], pred, proba)
    return {
        "model": model_name,
        "note": "含事后信息（破坏性性能测试），指标不可作为上线依据，仅用于量化泄漏影响",
        "n_features": int(X.shape[1]),
        "added_features": [c for c in X.columns if c not in cfg.MODEL_FEATURES],
        "threshold": threshold["threshold"],
        "test_metrics": metrics,
    }
