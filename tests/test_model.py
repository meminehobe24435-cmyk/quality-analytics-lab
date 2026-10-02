"""建模模块的测试。

覆盖：指标计算（含退化输入：全 0 / 全 1 / 无非正例预测）、阈值选择、
分层切分与互斥性、交叉验证可复现、特征重要性（含无原生重要性的模型）、
信息泄漏字段被排除、以及「测试集只评估一次」这一约定的机械化验证。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qalab import config as cfg
from qalab import model as m
from sklearn.linear_model import LogisticRegression


# --------------------------------------------------------------------------
# 指标计算
# --------------------------------------------------------------------------
def test_compute_metrics_normal_case():
    y_true = [0, 0, 1, 1, 0, 1, 0, 0]
    y_pred = [0, 0, 1, 0, 0, 1, 1, 0]
    proba = [0.1, 0.2, 0.8, 0.4, 0.3, 0.9, 0.6, 0.15]
    res = m.compute_metrics(y_true, y_pred, proba)
    assert res["n"] == 8
    assert res["n_positive"] == 3
    assert res["precision"] == pytest.approx(2 / 3)
    assert res["recall"] == pytest.approx(2 / 3)
    assert res["f1"] == pytest.approx(2 / 3)
    assert res["accuracy"] == pytest.approx(6 / 8)
    assert res["confusion_matrix"] == [[4, 1], [1, 2]]
    assert 0 < res["average_precision"] <= 1
    assert 0 < res["roc_auc"] <= 1
    assert res["single_class"] is False
    assert res["brier_score"] is not None


def test_compute_metrics_all_zero_labels_is_degenerate():
    """全 0 标签：AP 与 ROC-AUC 无定义，必须返回 None 而不是 1.0。"""
    res = m.compute_metrics([0, 0, 0, 0], [0, 0, 0, 0], [0.1, 0.2, 0.1, 0.3])
    assert res["single_class"] is True
    assert res["average_precision"] is None
    assert res["roc_auc"] is None
    assert res["brier_score"] is None
    assert res["balanced_accuracy"] is None
    assert res["accuracy"] == pytest.approx(1.0)
    assert res["specificity"] == pytest.approx(1.0)
    assert res["precision"] == 0.0
    assert res["recall"] == 0.0
    assert res["f1"] == 0.0


def test_compute_metrics_all_one_labels_is_degenerate():
    res = m.compute_metrics([1, 1, 1], [1, 1, 1], [0.9, 0.8, 0.95])
    assert res["single_class"] is True
    assert res["average_precision"] is None, "sklearn 对全 1 会给 1.0，具有误导性，这里返回 None"
    assert res["roc_auc"] is None
    assert res["recall"] == pytest.approx(1.0)
    assert res["accuracy"] == pytest.approx(1.0)


def test_compute_metrics_no_positive_prediction():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 0, 0]
    res = m.compute_metrics(y_true, y_pred, [0.1, 0.2, 0.3, 0.4])
    assert res["no_positive_prediction"] is True
    assert res["precision"] == 0.0
    assert res["recall"] == 0.0
    assert res["f1"] == 0.0
    assert res["confusion_matrix"] == [[2, 0], [2, 0]]
    # 概率是有的，所以排序类指标依然可以计算
    assert res["average_precision"] is not None
    assert res["roc_auc"] is not None


def test_compute_metrics_empty_input():
    res = m.compute_metrics([], [], None)
    assert res["n"] == 0
    assert res["average_precision"] is None
    assert res["confusion_matrix"] is None


def test_compute_metrics_without_probabilities():
    res = m.compute_metrics([0, 1], [0, 1], None)
    assert res["average_precision"] is None
    assert res["roc_auc"] is None
    assert res["f1"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# 阈值选择
# --------------------------------------------------------------------------
def test_select_threshold_maximises_f1():
    y_true = np.array([0, 0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8])
    res = m.select_threshold(y_true, proba)
    assert res["threshold"] == pytest.approx(0.7)
    assert res["score"] == pytest.approx(1.0)
    assert res["metric"] == "f1"


def test_select_threshold_is_deterministic():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = rng.random(200)
    a = m.select_threshold(y, p)
    b = m.select_threshold(y, p)
    assert a == b


def test_select_threshold_degenerate_falls_back_to_half():
    res = m.select_threshold([0, 0, 0], [0.1, 0.2, 0.3])
    assert res["threshold"] == 0.5
    assert res["score"] is None
    assert "回退" in res["note"]
    res2 = m.select_threshold([], [])
    assert res2["threshold"] == 0.5


def test_select_threshold_precision_metric():
    y_true = np.array([0, 0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8])
    res = m.select_threshold(y_true, proba, metric="precision")
    assert res["metric"] == "precision"
    assert res["score"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# 切分
# --------------------------------------------------------------------------
def _toy_frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.choice(["a", "b", "c"], size=n)
    z = -1.0 + 1.2 * x1 + (x2 == "c") * 0.8
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "machine_id": x2, "is_fail": y})


def test_split_data_is_disjoint_and_complete():
    df = _toy_frame()
    X, numeric, categorical = m.prepare_features(df)
    y = df[cfg.LABEL_COL]
    sp = m.split_data(X, y)
    tr, va, te = sp["idx_train"], sp["idx_val"], sp["idx_test"]
    assert len(set(tr) & set(va)) == 0
    assert len(set(tr) & set(te)) == 0
    assert len(set(va) & set(te)) == 0
    assert len(set(tr) | set(va) | set(te)) == len(X)
    assert len(te) == pytest.approx(len(X) * cfg.TEST_SIZE, abs=2)
    assert len(va) == pytest.approx(len(X) * cfg.TEST_SIZE * 0.8, abs=2)


def test_split_data_is_stratified():
    df = _toy_frame()
    X, _n, _c = m.prepare_features(df)
    y = df[cfg.LABEL_COL]
    sp = m.split_data(X, y)
    rates = [
        sp["y_train"].mean(),
        sp["y_val"].mean(),
        sp["y_test"].mean(),
    ]
    assert max(rates) - min(rates) < 0.06, f"三个集合的不合格率差异过大: {rates}"


def test_split_data_is_reproducible():
    df = _toy_frame()
    X, _n, _c = m.prepare_features(df)
    y = df[cfg.LABEL_COL]
    a = m.split_data(X, y)
    b = m.split_data(X, y)
    assert np.array_equal(a["idx_test"], b["idx_test"])
    assert a["y_test"].tolist() == b["y_test"].tolist()


# --------------------------------------------------------------------------
# 特征与泄漏
# --------------------------------------------------------------------------
def test_prepare_features_excludes_leaky_columns(clean_wide):
    X, numeric, categorical = m.prepare_features(clean_wide)
    for col in cfg.LEAKY_COLUMNS:
        assert col not in X.columns, f"{col} 是检验环节字段，不得进入特征"
    assert set(numeric) == set(cfg.MODEL_NUMERIC)
    assert set(categorical) == set(cfg.MODEL_CATEGORICAL)
    assert len(X.columns) == len(numeric) + len(categorical)


def test_prepare_features_with_perf_adds_missing_indicators(clean_wide):
    X, numeric, _c = m.prepare_features(clean_wide, include_perf=True)
    assert len(X.columns) > len(cfg.MODEL_FEATURES)
    for col in cfg.PERF_NUMERIC:
        assert col in X.columns
        assert f"{col}_was_missing" in X.columns
    assert X["tensile_strength_mpa_was_missing"].isin([0, 1]).all()


def test_leaky_feature_would_be_perfectly_predictive(clean_wide):
    """泄漏的现实后果：把缺陷类型放进特征，AP 会变成 1.0（开发中实测过）。

    这个测试把「为什么必须剔除检验环节字段」变成一条可执行、可复现的证据。
    """
    cols = list(cfg.MODEL_NUMERIC) + list(cfg.MODEL_CATEGORICAL) + ["defect_type"]
    X = clean_wide[cols].copy()
    y = clean_wide[cfg.LABEL_COL].astype(int)
    sp = m.split_data(X, y)
    model = m.build_pipeline(
        "logistic_regression", list(cfg.MODEL_NUMERIC), list(cfg.MODEL_CATEGORICAL) + ["defect_type"]
    )
    model.fit(sp["X_train"], sp["y_train"])
    proba = model.predict_proba(sp["X_test"])[:, 1]
    metrics = m.compute_metrics(sp["y_test"], (proba >= 0.5).astype(int), proba)
    assert metrics["average_precision"] > 0.99
    assert metrics["roc_auc"] > 0.99

    # 剔除该字段后，指标回到正常水平
    X2, num2, cat2 = m.prepare_features(clean_wide)
    y2 = clean_wide[cfg.LABEL_COL].astype(int)
    sp2 = m.split_data(X2, y2)
    model2 = m.build_pipeline("logistic_regression", num2, cat2)
    model2.fit(sp2["X_train"], sp2["y_train"])
    proba2 = model2.predict_proba(sp2["X_test"])[:, 1]
    m2 = m.compute_metrics(sp2["y_test"], (proba2 >= 0.5).astype(int), proba2)
    assert m2["average_precision"] < 0.9


# --------------------------------------------------------------------------
# 模型与交叉验证
# --------------------------------------------------------------------------
def test_build_models_contains_four_models_with_imbalance_handling():
    """四个模型都要处理类别不平衡：三个用 class_weight，神经网络用随机过采样。"""
    models = m.build_models(cfg.SEED)
    assert set(models) == {
        "logistic_regression",
        "random_forest",
        "hist_gradient_boosting",
        "mlp_classifier",
    }
    for name in ("logistic_regression", "random_forest", "hist_gradient_boosting"):
        clf = models[name].named_steps["clf"]
        assert getattr(clf, "class_weight", None) is not None, f"{name} 未处理类别不平衡"
    # MLPClassifier 在 sklearn 里没有 class_weight，必须用重采样包装
    mlp_step = models["mlp_classifier"].named_steps["clf"]
    assert isinstance(mlp_step, m.OversampledClassifier)
    assert not hasattr(mlp_step.base, "class_weight")


def test_oversampled_classifier_balances_training_data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})
    y = pd.Series([0] * 270 + [1] * 30)
    clf = m.OversampledClassifier(
        base=LogisticRegression(max_iter=200), random_state=cfg.SEED
    )
    clf.fit(X, y)
    assert clf.n_train_original_ == 300
    assert clf.n_train_resampled_ == 540  # 两类各 270
    assert clf.n_oversampled_added_ == 240
    # 预测阶段不做采样，输出行数与输入一致
    proba = clf.predict_proba(X)
    assert proba.shape == (300, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_oversampled_classifier_is_deterministic_and_leak_free():
    """过采样只发生在 fit 内部，且同一随机种子结果一致。"""
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.normal(size=200)})
    y = pd.Series([0] * 180 + [1] * 20)
    a = m.OversampledClassifier(base=LogisticRegression(), random_state=7).fit(X, y)
    b = m.OversampledClassifier(base=LogisticRegression(), random_state=7).fit(X, y)
    assert a.n_train_resampled_ == b.n_train_resampled_
    proba_a = a.predict_proba(X)
    proba_b = b.predict_proba(X)
    assert np.allclose(proba_a, proba_b)


def test_oversampled_classifier_single_class_does_not_crash():
    """退化输入：训练标签只有单一类别时退化为常数预测器（多数基学习器会直接报错）。"""
    X = pd.DataFrame({"a": [0.1, 0.2, 0.3, 0.4]})
    y = pd.Series([1, 1, 1, 1])
    clf = m.OversampledClassifier(base=LogisticRegression(), random_state=0).fit(X, y)
    assert clf.n_oversampled_added_ == 0
    assert clf.n_train_resampled_ == 4
    assert clf.single_class_ is True
    proba = clf.predict_proba(X)
    assert proba.shape == (4, 2), "接口必须保持 2 列，与其它模型一致"
    assert np.allclose(proba[:, 1], 1.0)
    assert (clf.predict(X) == 1).all()

    # 全 0 的标签同样要能处理
    clf0 = m.OversampledClassifier(base=LogisticRegression(), random_state=0).fit(
        X, pd.Series([0, 0, 0, 0])
    )
    assert np.allclose(clf0.predict_proba(X)[:, 0], 1.0)
    assert (clf0.predict(X) == 0).all()


def test_oversampled_classifier_works_in_cross_validation():
    """必须能在交叉验证里跑通：若标签识别（is_classifier）失效，roc_auc 会报形状错误。"""
    from sklearn.base import is_classifier

    df = _toy_frame(n=300, seed=3)
    X, numeric, categorical = m.prepare_features(df)
    y = df[cfg.LABEL_COL]
    from sklearn.model_selection import StratifiedKFold, cross_validate

    model = m.build_pipeline("mlp_classifier", numeric, categorical)
    assert is_classifier(model) is True
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=cfg.SEED)
    scores = cross_validate(model, X, y, cv=cv, scoring={"roc_auc": "roc_auc"}, n_jobs=1)
    assert len(scores["test_roc_auc"]) == 3
    assert all(0.0 <= s <= 1.0 for s in scores["test_roc_auc"])


def test_build_pipeline_unknown_model_raises():
    with pytest.raises(KeyError):
        m.build_pipeline("no_such_model", ["a"], ["b"])


def test_build_pipeline_fits_and_predicts(clean_wide):
    X, numeric, categorical = m.prepare_features(clean_wide)
    y = clean_wide[cfg.LABEL_COL].astype(int)
    sp = m.split_data(X, y)
    pipe = m.build_pipeline("logistic_regression", numeric, categorical)
    pipe.fit(sp["X_train"], sp["y_train"])
    proba = pipe.predict_proba(sp["X_test"])
    assert proba.shape == (len(sp["X_test"]), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_preprocessor_handles_unseen_categories(clean_wide):
    """线上会出现训练时没见过的类别（例如新机台），必须能预测而不是报错。"""
    X, numeric, categorical = m.prepare_features(clean_wide)
    y = clean_wide[cfg.LABEL_COL].astype(int)
    sp = m.split_data(X, y)
    pipe = m.build_pipeline("logistic_regression", numeric, categorical)
    pipe.fit(sp["X_train"], sp["y_train"])
    unseen = sp["X_test"].copy()
    unseen["machine_id"] = "M99-NEW"
    proba = pipe.predict_proba(unseen)
    assert proba.shape[0] == len(unseen)
    assert np.isfinite(proba).all()


def test_cross_validation_is_reproducible_and_has_all_metrics():
    df = _toy_frame()
    X, numeric, categorical = m.prepare_features(df)
    y = df[cfg.LABEL_COL]
    a = m.cross_validate_models(X, y, numeric, categorical, seed=cfg.SEED, folds=3)
    b = m.cross_validate_models(X, y, numeric, categorical, seed=cfg.SEED, folds=3)
    assert a == b
    for name in a:
        for metric in ("average_precision", "roc_auc", "f1", "precision", "recall"):
            assert metric in a[name]
            assert len(a[name][metric]["folds"]) == 3
            assert 0 <= a[name][metric]["mean"] <= 1
            assert a[name][metric]["std"] >= 0


def test_extract_feature_importance_for_linear_and_tree(clean_wide):
    X, numeric, categorical = m.prepare_features(clean_wide)
    y = clean_wide[cfg.LABEL_COL].astype(int)
    sp = m.split_data(X, y)

    lin = m.build_pipeline("logistic_regression", numeric, categorical)
    lin.fit(sp["X_train"], sp["y_train"])
    imp = m.extract_feature_importance(lin, top_k=5)
    assert len(imp) == 5
    assert imp[0]["importance_source"] == "standardized_coefficient"
    assert all("direction" in item for item in imp)
    assert imp == sorted(imp, key=lambda d: -d["importance"])

    tree = m.build_pipeline("random_forest", numeric, categorical)
    tree.fit(sp["X_train"], sp["y_train"])
    imp_tree = m.extract_feature_importance(tree, top_k=5)
    assert imp_tree[0]["importance_source"] == "native_importance"


def test_extract_feature_importance_uses_permutation_for_hist_gb(clean_wide):
    """HistGradientBoosting 在 sklearn 里没有原生重要性，必须回退到置换重要性。"""
    X, numeric, categorical = m.prepare_features(clean_wide)
    y = clean_wide[cfg.LABEL_COL].astype(int)
    sp = m.split_data(X, y)
    model = m.build_pipeline("hist_gradient_boosting", numeric, categorical)
    model.fit(sp["X_train"], sp["y_train"])
    # 不传验证集 -> 无原生重要性可以回退，返回空列表而不是抛异常
    assert m.extract_feature_importance(model) == []
    imp = m.extract_feature_importance(model, X_val=sp["X_val"], y_val=sp["y_val"])
    assert len(imp) > 0
    assert imp[0]["importance_source"] == "permutation_importance_on_validation"
    assert imp[0]["importance"] > 0


# --------------------------------------------------------------------------
# 完整建模流程
# --------------------------------------------------------------------------
def test_run_modeling_returns_expected_structure(modeling):
    result, extras = modeling
    assert result["n_samples"] == len(extras["y_all"])
    assert set(result["models"]) == {
        "logistic_regression",
        "random_forest",
        "hist_gradient_boosting",
        "mlp_classifier",
    }
    assert result["best_model_by_validation_ap"] in result["models"]
    for col in cfg.LEAKY_COLUMNS:
        assert col not in result["feature_columns"]
    for name, res in result["models"].items():
        assert 0 < res["threshold_selection"]["threshold"] < 1
        tm = res["test_metrics"]
        assert 0 <= tm["average_precision"] <= 1
        assert 0 <= tm["roc_auc"] <= 1
        assert len(tm["confusion_matrix"]) == 2
        assert len(res["top_features"]) > 0
    assert set(extras["test_proba"]) == set(result["models"])


def test_test_set_is_evaluated_with_validation_threshold_only(modeling):
    """机械化验证「阈值只在验证集上选、测试集只评估一次」这一约定。

    做法：独立地用验证集概率重算阈值，必须与流程返回的阈值完全一致；
    并用该阈值在测试集上重算指标，必须与流程报告的一致。
    """
    result, extras = modeling
    y_val = pd.Series(extras["y_val"])
    y_test = pd.Series(extras["y_test"])
    for name, res in result["models"].items():
        expected = m.select_threshold(y_val, extras["val_proba"][name])
        assert res["threshold_selection"]["threshold"] == expected["threshold"]
        recomputed = m.compute_metrics(
            y_test,
            (extras["test_proba"][name] >= expected["threshold"]).astype(int),
            extras["test_proba"][name],
        )
        assert recomputed["f1"] == res["test_metrics"]["f1"]
        assert recomputed["average_precision"] == res["test_metrics"]["average_precision"]


def test_run_modeling_is_reproducible(clean_wide):
    a, _ = m.run_modeling(clean_wide, with_perf_ablation=False)
    b, _ = m.run_modeling(clean_wide, with_perf_ablation=False)
    assert a["best_model_by_validation_ap"] == b["best_model_by_validation_ap"]
    assert a["models"] == b["models"]
    assert a["cv"] == b["cv"]


def test_run_modeling_ablation_marks_post_hoc_features(modeling):
    result, _ = modeling
    abl = result["ablation_with_perf_features"]
    assert abl["n_features"] > result["n_features"]
    assert set(abl["added_features"]).issuperset(set(cfg.PERF_NUMERIC))
    assert "不可" in abl["note"] or "事后" in abl["note"]


def test_run_modeling_output_is_json_serializable(modeling):
    import json

    result, _ = modeling
    text = json.dumps(result, ensure_ascii=False)
    assert "average_precision" in text


def test_run_modeling_beats_random_baseline(modeling):
    """模型必须明显优于随机：AP 至少达到基线不合格率的 3 倍，ROC-AUC > 0.7。

    （基线 = 随机打分时 AP 约等于正例比例约 0.12。）
    """
    result, _ = modeling
    best = result["best_model_by_validation_ap"]
    tm = result["models"][best]["test_metrics"]
    base = tm["positive_rate"]
    assert tm["average_precision"] > 3 * base
    assert tm["roc_auc"] > 0.7
