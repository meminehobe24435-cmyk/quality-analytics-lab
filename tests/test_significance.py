"""显著因子识别的测试。

最关键的一项：**注入已知信号，断言因子被正确识别出来**。
这里有两层验证：
  1. 手工构造一个「只有一个因子有强效应 + 两个诱饵因子」的数据集，
     断言有效应因子被识别、诱饵因子不被识别；
  2. 用项目自己的合成数据生成机理（真值已知）跨多个种子验证：
     设计的主因子应被稳定识别，真值零效应的诱饵因子不应被稳定误判。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qalab import config as cfg
from qalab import io as qio
from qalab import significance as sg
from qalab import synth


# --------------------------------------------------------------------------
# 手工构造的已知信号
# --------------------------------------------------------------------------
def _injected_signal_frame(n_runs: int = 120, units_per_run: int = 20, seed: int = 2024):
    """构造已知真值的数据：只有 `driver` 影响标签，`decoy_a`/`decoy_b` 无影响。

    因子都在炉次级变化（簇内常数），因此在炉次级汇总口径上检验。
    返回 (单元级表, 炉次级汇总表)，后者按 `build_run_level_table` 的契约
    带上 fail_rate / n_units / n_fail。
    """
    rng = np.random.default_rng(seed)
    rows = []
    run_rows = []
    for r in range(n_runs):
        driver = rng.normal(0, 1)
        decoy_a = rng.normal(0, 1)
        decoy_b = str(rng.choice(["X", "Y", "Z"]))
        z = -1.6 + 1.9 * driver  # 只有 driver 进入潜在风险
        p = 1 / (1 + np.exp(-z))
        n_fail = 0
        for u in range(units_per_run):
            flag = int(rng.random() < p)
            n_fail += flag
            rows.append(
                {
                    "unit_id": f"U{r:04d}{u:03d}",
                    "run_id": f"R{r:04d}",
                    "new_driver": driver,
                    "decoy_numeric": decoy_a,
                    "decoy_category": decoy_b,
                    "is_fail": flag,
                }
            )
        run_rows.append(
            {
                "run_id": f"R{r:04d}",
                "fail_rate": n_fail / units_per_run,
                "n_units": units_per_run,
                "n_fail": n_fail,
                "new_driver": driver,
                "decoy_numeric": decoy_a,
                "decoy_category": decoy_b,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(run_rows)



@pytest.fixture(scope="module")
def injected():
    """在手工构造的已知真值数据上跑完整筛选（不需要 qalab 的列分组配置）。"""
    unit_df, run_df = _injected_signal_frame()
    tests = sg.screen_factors(
        unit_df,
        run_df,
        None,
        factors=["new_driver", "decoy_numeric", "decoy_category"],
        n_perm=300,
    )
    return unit_df, run_df, tests, sg.rank_factors(tests)


def test_injected_signal_is_identified(injected):
    """注入已知信号 -> 该因子必须被识别为显著，诱饵因子必须不显著。"""
    _unit_df, _run_df, _tests, ranking = injected
    by_factor = ranking.set_index("factor")

    assert bool(by_factor.loc["new_driver", "factor_significant"]) is True
    assert bool(by_factor.loc["new_driver", "unit_significant"]) is True
    assert float(by_factor.loc["new_driver", "factor_q_value"]) < 0.01
    assert float(by_factor.loc["new_driver", "factor_effect_size"]) > 0.3
    assert float(by_factor.loc["new_driver", "factor_statistic"]) > 0.5
    assert by_factor.loc["new_driver", "factor_level"] == "run"
    assert by_factor.loc["new_driver", "n_independent"] == 120

    # 诱饵：真值零效应，不应被判为显著
    assert bool(by_factor.loc["decoy_numeric", "factor_significant"]) is False
    assert bool(by_factor.loc["decoy_category", "factor_significant"]) is False
    # 排序：真信号必须排第一
    assert ranking.iloc[0]["factor"] == "new_driver"


def test_injected_categorical_decoy_is_treated_as_categorical(injected):
    """手工构造的字符串列必须被识别为类别因子（按 dtype 兜底）。"""
    _unit_df, _run_df, tests, _ranking = injected
    kinds = tests.groupby("factor")["kind"].first().to_dict()
    assert kinds["decoy_category"] == "categorical"
    assert kinds["decoy_numeric"] == "numeric"
    assert kinds["new_driver"] == "numeric"
    cat_method = tests[
        (tests["factor"] == "decoy_category") & (tests["level"] == sg.LEVEL_NAIVE)
    ].iloc[0]["method"]
    assert cat_method == "chi_square"



def test_injected_signal_effect_direction_is_positive(injected):
    """方向也要对：driver 越大越容易不合格，效应量必须为正。"""
    _unit_df, _run_df, tests, _ranking = injected
    row = tests[(tests["factor"] == "new_driver") & (tests["level"] == sg.LEVEL_FACTOR)].iloc[0]
    assert row["effect_size"] > 0
    assert row["method"] == "spearman_vs_fail_rate"


def test_screening_table_structure(injected):
    _unit_df, _run_df, tests, ranking = injected
    assert set(tests["level"]) == {sg.LEVEL_NAIVE, sg.LEVEL_CLUSTER, sg.LEVEL_FACTOR}
    assert set(tests["role"]) == {sg.ROLE_PRIMARY, sg.ROLE_SUPPLEMENTARY}
    for col in (
        "factor",
        "kind",
        "level",
        "method",
        "statistic",
        "p_value",
        "q_value",
        "effect_size",
        "mutual_info",
        "significant",
    ):
        assert col in tests.columns
    # 每个因子在每层只应有一个主口径结果
    primary = tests[tests["role"] == sg.ROLE_PRIMARY]
    assert not primary.duplicated(subset=["factor", "level"]).any()
    # 排序表包含所有因子
    assert set(ranking["factor"]) == {"new_driver", "decoy_numeric", "decoy_category"}
    assert ranking["rank"].tolist() == [1, 2, 3]


def test_naive_unit_test_is_anti_conservative_on_clustered_data(injected):
    """朴素单元级检验在簇相关数据上会给出比簇置换小得多的 p 值（伪重复）。"""
    _unit_df, _run_df, tests, _ranking = injected
    for factor in ("new_driver", "decoy_numeric"):
        naive = tests[
            (tests["factor"] == factor)
            & (tests["level"] == sg.LEVEL_NAIVE)
            & (tests["role"] == sg.ROLE_PRIMARY)
        ].iloc[0]
        cluster = tests[
            (tests["factor"] == factor) & (tests["level"] == sg.LEVEL_CLUSTER)
        ].iloc[0]
        assert naive["p_value"] < cluster["p_value"], f"{factor} 的朴素 p 值应更小"


def test_factor_level_assignment_uses_batch_for_material_factors(
    clean_wide, run_level, batch_level
):
    tests = sg.screen_factors(
        clean_wide,
        run_level,
        batch_level,
        factors=["material_moisture_pct", "furnace_temp_c"],
        n_perm=50,
    )
    own = tests[tests["level"] == sg.LEVEL_FACTOR].set_index("factor")
    assert own.loc["material_moisture_pct", "level_detail"] == "batch"
    assert own.loc["furnace_temp_c", "level_detail"] == "run"
    # 层级对应的独立观测数
    assert own.loc["material_moisture_pct", "n_used"] == len(batch_level)
    assert own.loc["furnace_temp_c", "n_used"] == len(run_level)
    assert sg.cluster_key_for_factor("material_supplier") == "batch_id"
    assert sg.cluster_key_for_factor("machine_id") == "run_id"


def test_mutual_information_is_deterministic(clean_wide):
    features = ["furnace_temp_c", "machine_id", "pressure_mpa"]
    a = sg.mutual_information_scores(clean_wide, features, seed=cfg.SEED)
    b = sg.mutual_information_scores(clean_wide, features, seed=cfg.SEED)
    assert a == b
    assert set(a) == set(features)
    assert all(v is not None and v >= 0 for v in a.values())


def test_mutual_information_degenerate_inputs():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "is_fail": [1, 1, 1]})
    scores = sg.mutual_information_scores(df, ["x"])
    assert scores == {"x": None}
    assert sg.mutual_information_scores(pd.DataFrame({"is_fail": []}), ["x"]) == {"x": None}


def test_encode_for_mutual_info_handles_categorical():
    df = pd.DataFrame({"num": [1.0, 2.0, 3.0], "cat": ["a", "b", "a"]})
    encoded, discrete = sg.encode_for_mutual_info(df, ["num", "cat"])
    assert list(encoded.columns) == ["num", "cat"]
    assert discrete == [False, True]
    assert encoded["cat"].nunique() == 2
    assert encoded["num"].isna().sum() == 0


def test_screen_factors_on_empty_frame():
    empty = pd.DataFrame({"is_fail": []})
    tests = sg.screen_factors(empty, pd.DataFrame(), pd.DataFrame(), factors=[])
    assert tests.empty
    ranking = sg.rank_factors(tests)
    assert ranking.empty
    assert sg.significant_factors(ranking) == []


def test_rank_factors_single_factor_with_all_nan_pvalue():
    tests = pd.DataFrame(
        [
            {
                "level": sg.LEVEL_FACTOR,
                "level_detail": "run",
                "factor": "x",
                "kind": "numeric",
                "method": "spearman_vs_fail_rate",
                "role": sg.ROLE_PRIMARY,
                "statistic": None,
                "p_value": np.nan,
                "q_value": np.nan,
                "effect_size": None,
                "effect_metric": "spearman_rho",
                "n_used": 0,
                "significant": False,
                "mutual_info": None,
            }
        ]
    )
    ranking = sg.rank_factors(tests)
    assert len(ranking) == 1
    assert ranking.iloc[0]["verdict"] == "不显著"
    assert sg.significant_factors(ranking, "factor") == []


# --------------------------------------------------------------------------
# 跨种子稳定性：真值来自生成机理
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def multiseed_results():
    """在 5 个种子上跑「数据生成 + 合并 + 筛选」，统计显著性结论的稳定性。"""
    seeds = [cfg.SEED + i * 1000 for i in range(5)]
    frames = []
    for seed in seeds:
        tables, manifest = synth.generate_all(seed)
        wide = qio.build_analysis_table(tables)
        clean, _ = qio.clean_analysis_table(wide)
        tests = sg.screen_factors(
            clean,
            qio.build_run_level_table(clean),
            qio.build_batch_level_table(clean),
            seed=seed,
            n_perm=1000,
        )
        ranking = sg.rank_factors(tests)
        ranking = ranking.assign(seed=seed)
        frames.append(ranking)
    return pd.concat(frames, ignore_index=True), manifest, len(seeds)


def test_ground_truth_is_documented(multiseed_results):
    _allr, manifest, _n = multiseed_results
    truth = manifest["ground_truth_mechanism"]
    assert set(truth["primary_factors"]) == {
        "furnace_temp_c",
        "material_moisture_pct",
        "pressure_mpa",
    }
    assert "ambient_humidity_pct" in truth["decoy_factors"]
    assert "operator_id" in truth["decoy_factors"]


def test_primary_factors_are_recovered_across_seeds(multiseed_results):
    """设计的主因子必须在多数种子上被识别为显著（因子层级口径）。

    实测（5 个种子，n_perm=1000）：
      furnace_temp_c        5/5
      pressure_mpa          5/5
      material_moisture_pct 3/5   <- 批次级因子只有 36 个独立观测，功效确实有限
    因此阈值取「至少 3/5」，并在 README 里如实写明含水率不是每次都显著 ——
    这属于真实的统计功效限制，不做人为调参掩盖。
    """
    allr, manifest, n_seeds = multiseed_results
    truth = manifest["ground_truth_mechanism"]
    counts = (
        allr.groupby("factor")["factor_significant"].sum().reindex(
            truth["primary_factors"]
        )
    )
    assert counts.notna().all()
    for factor, count in counts.items():
        assert count >= 3, f"{factor} 只在 {count}/{n_seeds} 个种子上显著"


def test_furnace_temp_and_pressure_are_always_recovered(multiseed_results):
    """两个最强的主因子必须每个种子都被识别（这是「注入信号能被稳定检出」的硬证据）。"""
    allr, _manifest, n_seeds = multiseed_results
    for factor in ("furnace_temp_c", "pressure_mpa"):
        count = int(allr.loc[allr["factor"] == factor, "factor_significant"].sum())
        assert count == n_seeds, f"{factor} 只在 {count}/{n_seeds} 个种子上显著"


def test_decoy_factors_are_not_recovered_across_seeds(multiseed_results):
    """真值零效应的诱饵因子不应被稳定误判（假阳性控制）。

    实测（5 个种子）：ambient_humidity_pct 0/5、material_particle_size_um 0/5、
    operator_id 0/5（其中 operator_id 有一个种子的 q=0.052，刚好在门线之上）。
    这里允许最多 1/5，容忍单次抽样波动。
    """
    allr, manifest, n_seeds = multiseed_results
    truth = manifest["ground_truth_mechanism"]
    for factor in truth["decoy_factors"]:
        count = int(allr.loc[allr["factor"] == factor, "factor_significant"].sum())
        assert count <= 1, f"诱饵因子 {factor} 在 {count}/{n_seeds} 个种子上被误判为显著"


def test_naive_level_is_systematically_more_anti_conservative(multiseed_results):
    """朴素单元级检验在跨种子平均下判出的显著因子数，应明显多于有效口径。"""
    allr, _manifest, _n = multiseed_results
    naive = int(allr.groupby("seed")["unit_naive_significant"].sum().mean())
    valid = int(allr.groupby("seed")["factor_significant"].sum().mean())
    assert naive > valid, f"朴素口径平均显著 {naive} 个，有效口径 {valid} 个，前者应更多"
