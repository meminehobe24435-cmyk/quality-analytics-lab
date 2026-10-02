"""合成数据生成：4 张表，带物理动机的失效机理。

⚠️ 数据是【合成】的，不是真实产线数据。生成过程写成一个显式的潜在失效机理：
存在一个炉次级潜在风险分 z（越大越容易不合格），由炉温偏离最优值、原料含水率、
压力、线速、冷却速率、机台、班次共同决定；单元级再叠加随机波动决定该单元
是否被判不合格。这样「显著因子识别」与「预测模型」才有可验证的真值——
测试可以断言注入的信号确实被识别出来。

生成的关系（真值，用于校准显著性结论）：
  1. 炉温偏离 1505℃ 越高风险越大，过热（正偏离）比欠热更危险 —— 主效应
  2. 原料含水率越高风险越大（批次效应，供应商相关） —— 主效应
  3. 压力、线速偏高增加风险；冷却速率偏高降低风险 —— 次级效应
  4. 机台 M3、班次 C 显著更差 —— 类别主效应
  5. 环境湿度、检验员、供应商 —— 弱效应/诱饵因子，BH 校正后应不显著
  6. 性能测试指标（拉伸强度等）由同一潜在 z 派生，因此与标签强相关，
     但它们是【检验后】才测到的破坏性试验结果，不能进预测模型
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg

# 生成机理使用的系数（标准化后的线性贡献，便于解释量级）
# 校准说明：系数大小经过校准，使得在 96 个炉次（= 96 个独立观测）的样本量下，
# 「设计的主效应」能被单因子筛选稳定识别出来，而「设计的零效应因子（诱饵）」
# 不会稳定被判为显著。跨 5 个种子的稳定性在 tests/test_significance.py 中断言。
_TEMP_OPTIMUM = 1505.0
_TEMP_SD = 8.5
_MOIST_SD = 0.45
_PRES_SD = 0.26
_SPEED_SD = 4.20
_COOL_SD = 1.38
_HUM_SD = 12.0

_TEMP_COEF_OVER = 1.55
"""过热（温度高于最优值）每一标准差的 logit 贡献。"""

_TEMP_COEF_UNDER = 0.70
"""欠热每一标准差的 logit 贡献（比过热温和，但仍增险）。"""

_MOISTURE_COEF = 1.35
_PRESSURE_COEF = 1.05
_SPEED_COEF = 0.35
_COOLING_COEF = -0.40
_HUMIDITY_COEF = 0.0
"""环境湿度：**真值为零效应**的诱饵因子，用于检验假阳性控制。"""

_BASE_LOGIT = -3.95
"""基准 logit。校准到整体不合格率约 15%（见 tools/calibrate.py 的输出）。"""
_MACHINE_EFFECT = {"M1": 0.00, "M2": 0.15, "M3": 1.75, "M4": 0.05}
"""机台效应：M3 是已知的问题机台（真实缺陷率明显更高）。"""
_SHIFT_EFFECT = {"A": 0.00, "B": 0.20, "C": 0.70}
_SUPPLIER_MOISTURE_SHIFT = {"SUP-A": 0.00, "SUP-B": 0.26, "SUP-C": 0.58}
# 检验员：真值为零效应（诱饵）。真实工厂里常怀疑检验员"手法松严"，但这里真值无效应
_OPERATOR_EFFECT = {"OP03": 0.0, "OP09": 0.0}
_RUN_NOISE_SD = 0.20
_UNIT_NOISE_SD = 0.28


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _timestamps(rng: np.random.Generator, n: int) -> pd.DatetimeIndex:
    """从固定起始时间按固定节拍生成时间戳，保证可复现。"""
    base = np.datetime64("2025-03-03T06:00:00")
    steps = np.arange(n) * np.timedelta64(4, "h") + rng.integers(0, 90, size=n).astype(
        "timedelta64[m]"
    )
    return pd.DatetimeIndex(base + steps)


def generate_material_batches(seed: int = cfg.SEED) -> pd.DataFrame:
    """原料批次主数据：供应商 + 批次级含水率（批次效应的来源）。"""
    rng = np.random.default_rng(seed)
    batch_ids = [f"B{i:02d}" for i in range(1, cfg.N_BATCHES + 1)]
    suppliers = rng.choice(
        ["SUP-A", "SUP-B", "SUP-C"], size=cfg.N_BATCHES, p=[0.45, 0.35, 0.20]
    )
    moisture = np.array([_SUPPLIER_MOISTURE_SHIFT[s] for s in suppliers]) + 3.05 + rng.normal(
        0.0, 0.28, size=cfg.N_BATCHES
    )
    particle = rng.normal(62.0, 9.0, size=cfg.N_BATCHES)
    received = pd.to_datetime("2025-02-20") + pd.to_timedelta(
        rng.integers(0, 10, size=cfg.N_BATCHES), unit="D"
    )
    return pd.DataFrame(
        {
            "batch_id": batch_ids,
            "material_supplier": pd.Series(suppliers).astype("string"),
            "material_moisture_pct": np.round(moisture, 3),
            "material_particle_size_um": np.round(particle, 2),
            "received_date": received,
        }
    )


def generate_production_runs(
    batches: pd.DataFrame, seed: int = cfg.SEED + 1
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """生产参数（一炉一行）。

    返回值第二项是**内部**的潜在风险表（run_id -> z_run / p_run），
    仅用于在同一进程内派生检验与性能测试结果，不落盘（不进分析宽表）。
    """
    rng = np.random.default_rng(seed)
    n_runs = cfg.N_BATCHES * cfg.RUNS_PER_BATCH
    batch_id = np.repeat(batches["batch_id"].to_numpy(), cfg.RUNS_PER_BATCH)
    run_id = [f"R{i:03d}" for i in range(1, n_runs + 1)]

    machine_id = rng.choice(list(_MACHINE_EFFECT), size=n_runs, p=[0.28, 0.26, 0.24, 0.22])
    operator_pool = [f"OP{i:02d}" for i in range(1, 13)]
    operator_id = rng.choice(operator_pool, size=n_runs)

    furnace_temp = _TEMP_OPTIMUM + rng.normal(0.0, _TEMP_SD, size=n_runs)
    pressure = 3.00 + rng.normal(0.0, _PRES_SD, size=n_runs)
    line_speed = 21.0 + rng.normal(0.0, _SPEED_SD, size=n_runs)
    cooling_rate = 8.00 + rng.normal(0.0, _COOL_SD, size=n_runs)
    humidity = np.clip(rng.normal(48.0, _HUM_SD, size=n_runs), 22.0, 88.0)

    moisture = (
        batches.set_index("batch_id")["material_moisture_pct"].reindex(batch_id).to_numpy()
    )

    start_time = _timestamps(rng, n_runs)
    # 班次由排产时间决定（A=早班 06-13，B=中班 14-21，C=夜班 22-05），
    # 与炉温等工艺参数相互独立 -> 班次效应是干净的类别主效应，不会与温度混淆
    shift = np.array(["A", "B", "C"])[((start_time.hour - 6) % 24 // 8) % 3]
    shift_effect = np.array([_SHIFT_EFFECT[s] for s in shift])

    dev = furnace_temp - _TEMP_OPTIMUM
    # 最优炉温 1505℃：温度越高越危险，超过最优值后恶化更快。
    #
    # ⚠️ 开发中实测踩坑：最初写成「V 形」（偏离最优值即增险，左右对称），
    # 结果炉次级的口径是 Spearman 秩相关，而 V 形属于**非单调**效应，
    # 两支几乎互相抵消 —— 真正的头号因子（炉温）在 5 个种子里只有 2 个被检出。
    # 改成单调递增（过热更陡）之后才 5/5 稳定检出。教训：口径要匹配效应的形状。
    temp_term = (
        _TEMP_COEF_UNDER * dev / _TEMP_SD
        + (_TEMP_COEF_OVER - _TEMP_COEF_UNDER) * np.maximum(dev, 0.0) / _TEMP_SD
    )
    z = (
        _BASE_LOGIT
        + temp_term
        + _MOISTURE_COEF * (moisture - 3.43) / _MOIST_SD
        + _PRESSURE_COEF * (pressure - 3.00) / _PRES_SD
        + _SPEED_COEF * (line_speed - 21.0) / _SPEED_SD
        + _COOLING_COEF * (cooling_rate - 8.0) / _COOL_SD
        + _HUMIDITY_COEF * (humidity - 48.0) / _HUM_SD
        + np.array([_MACHINE_EFFECT[m] for m in machine_id])
        + shift_effect
        + np.array([_OPERATOR_EFFECT.get(o, 0.0) for o in operator_id])
        + rng.normal(0.0, _RUN_NOISE_SD, size=n_runs)
    )

    runs = pd.DataFrame(
        {
            "run_id": run_id,
            "batch_id": batch_id,
            "start_time": start_time,
            "machine_id": machine_id,
            "shift": pd.Series(shift).astype("string"),
            "operator_id": pd.Series(operator_id).astype("string"),
            "furnace_temp_c": np.round(furnace_temp, 2),
            "pressure_mpa": np.round(pressure, 3),
            "line_speed_mpm": np.round(line_speed, 2),
            "cooling_rate_cps": np.round(cooling_rate, 2),
            "ambient_humidity_pct": np.round(humidity, 1),
        }
    )
    latent = pd.DataFrame({"run_id": run_id, "z_run": z})
    return runs, latent


def generate_inspection_results(
    runs: pd.DataFrame, latent: pd.DataFrame, seed: int = cfg.SEED + 2
) -> pd.DataFrame:
    """质量检验结果：单元级的合格/不合格标签 + 缺陷类型 + 检验员/班次。

    缺陷类型按「失效机理」加权生成，使缺陷分布与根因可解释（而不是随机贴标签）。
    """
    rng = np.random.default_rng(seed)
    z_map = latent.set_index("run_id")["z_run"]

    rows: list[dict[str, object]] = []
    unit_no = 0
    # 缺陷类型 -> 最相关的因子（按机理分配概率）
    defect_pool = ["PORE", "CRACK", "DIMENSION", "CONTAMINATION", "EDGE"]
    defect_weight = np.array([0.36, 0.18, 0.22, 0.13, 0.11])
    inspectors = [f"INS{i:02d}" for i in range(1, 9)]
    methods = ["VISUAL", "XRAY", "DIMENSION"]

    for _, run in runs.iterrows():
        z_run = float(z_map.loc[run["run_id"]])
        n_units = int(rng.integers(cfg.UNITS_PER_RUN_MIN, cfg.UNITS_PER_RUN_MAX))
        unit_ids = [f"U{unit_no + i:05d}" for i in range(1, n_units + 1)]
        unit_no += n_units
        unit_z = z_run + rng.normal(0.0, _UNIT_NOISE_SD, size=n_units)
        p = _sigmoid(unit_z)
        is_fail = rng.random(n_units) < p
        for uid, flag in zip(unit_ids, is_fail):
            if flag:
                defect = str(rng.choice(defect_pool, p=defect_weight))
            else:
                defect = "NONE"
            rows.append(
                {
                    "unit_id": uid,
                    "run_id": run["run_id"],
                    "batch_id": run["batch_id"],
                    "shift": run["shift"],
                    "inspection_method": str(rng.choice(methods, p=[0.6, 0.28, 0.12])),
                    "inspector_id": str(rng.choice(inspectors)),
                    "defect_type": defect,
                    "is_fail": int(flag),
                }
            )

    df = pd.DataFrame(rows)
    ts = pd.to_datetime(runs["start_time"]).to_numpy() + np.timedelta64(6, "h")
    ts_map = dict(zip(runs["run_id"], ts))
    df["inspection_ts"] = pd.to_datetime(df["run_id"].map(ts_map))
    return df


def generate_perf_tests(
    inspections: pd.DataFrame, latent: pd.DataFrame, seed: int = cfg.SEED + 3
) -> pd.DataFrame:
    """性能测试（破坏性抽检）：与潜在风险同源，因此与标签相关但**事后可得**。

    单元级抽检 70%，部分单元复测 2~3 次 -> 与检验表构成一对多关系。
    数值列注入约 6% 缺失。

    性能指标以中心化后的潜在风险 z 为自变量（zc = z - mean(z)），
    这样「健康工艺」正好落在规格中心附近，规格上下限按 ±4.5 个标准差留边，
    越界值只来自真正的异常（传感器故障 / 工艺失控），而不是生成分布没校准好。
    """
    rng = np.random.default_rng(seed)
    z_map = latent.set_index("run_id")["z_run"]
    z_mean = float(latent["z_run"].mean())

    n_units = len(inspections)
    take = rng.random(n_units) < cfg.PERF_TEST_UNIT_FRACTION
    sampled = inspections.loc[take, ["unit_id", "run_id"]].copy()

    repeats = rng.choice(cfg.PERF_REPEAT_CHOICES, size=len(sampled))
    rows: list[dict[str, object]] = []
    for (unit_id, run_id), n_rep in zip(
        sampled[["unit_id", "run_id"]].to_numpy(), repeats
    ):
        zc = float(z_map.loc[run_id]) - z_mean
        for rep in range(1, int(n_rep) + 1):
            rows.append(
                {
                    "unit_id": unit_id,
                    "run_id": run_id,
                    "repeat_no": rep,
                    # 潜在风险越高（工艺越差）-> 强度/硬度/延伸率越低、表面越粗糙。
                    # 系数经过校准：过程能力 Cpk 约 1.33（规格宽度约 8σ），
                    # 因此正常情况下几乎不产生越界值，越界只会来自真正的异常。
                    "tensile_strength_mpa": 430.0 - 17.0 * zc + rng.normal(0, 8.0),
                    "hardness_hv": 120.0 - 3.00 * zc + rng.normal(0, 3.5),
                    "elongation_pct": 9.50 - 0.95 * zc + rng.normal(0, 1.00),
                    "surface_roughness_ra": 1.35 + 0.15 * zc + rng.normal(0, 0.12),
                }
            )
    df = pd.DataFrame(rows)
    for col in cfg.PERF_NUMERIC:
        df[col] = np.round(df[col], 3)
        df.loc[rng.random(len(df)) < cfg.PERF_MISSING_RATE, col] = np.nan
    df["test_ts"] = pd.to_datetime("2025-04-01") + pd.to_timedelta(
        rng.integers(0, 20000, size=len(df)), unit="m"
    )
    return df


def inject_known_anomalies(
    runs: pd.DataFrame,
    inspections: pd.DataFrame,
    perf_tests: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """注入**已知数量**的脏数据，用于验证质量检查确实抓得到。

    返回注入后的三张表 + 注入清单（清单会写进 metrics.json，便于对照）。
    """
    rng = np.random.default_rng(cfg.SEED + 9)
    runs = runs.copy()
    inspections = inspections.copy()
    perf_tests = perf_tests.copy()

    # 1) 传感器故障：若干炉次炉温 = 9999.0（远超规格上限）
    hit = runs.index[runs["run_id"].isin(
        [f"R{i:03d}" for i in cfg.INJECTED_OUT_OF_RANGE_RUNS]
    )]
    runs.loc[hit, "furnace_temp_c"] = cfg.INJECTED_OUT_OF_RANGE_VALUE

    # 2) 检验员缺失
    miss_idx = rng.choice(inspections.index, size=cfg.N_INJECTED_MISSING_INSPECTOR, replace=False)
    inspections.loc[miss_idx, "inspector_id"] = pd.NA

    # 3) 机台缺失
    miss_run_idx = rng.choice(runs.index, size=cfg.N_INJECTED_MISSING_MACHINE, replace=False)
    runs.loc[miss_run_idx, "machine_id"] = pd.NA

    # 4) 常量列 / 近常量列（真实系统里常见：单厂区编号、几乎全 False 的标记位）
    inspections[cfg.CONSTANT_COLUMN_NAME] = cfg.CONSTANT_COLUMN_VALUE
    n_archived = max(1, int(round(len(inspections) * cfg.NEAR_CONSTANT_TRUE_RATE)))
    archived = np.zeros(len(inspections), dtype=bool)
    archived[:n_archived] = True
    inspections[cfg.NEAR_CONSTANT_COLUMN_NAME] = archived

    # 5) 完全重复行
    dup_rows = inspections.iloc[list(rng.choice(len(inspections), size=cfg.N_INJECTED_DUPLICATE_ROWS, replace=False))]
    inspections = pd.concat([inspections, dup_rows], ignore_index=True)

    # 6) 某个单元复测到 3 次（一对多扇出）
    extra_src = perf_tests[perf_tests["repeat_no"] == 1].head(cfg.N_INJECTED_EXTRA_PERF_TESTS)
    extra = extra_src.copy()
    extra["repeat_no"] = 3
    perf_tests.loc[perf_tests["unit_id"].isin(extra["unit_id"]) & (perf_tests["repeat_no"] == 2), "repeat_no"] = 4
    perf_tests = pd.concat([perf_tests, extra], ignore_index=True)

    # 7) 真实越界：若干条拉伸强度低于规格下限（质量逃逸，混在正常抽检中）
    oos_idx = rng.choice(perf_tests.index, size=cfg.N_INJECTED_OOS_PERF, replace=False)
    low, high = cfg.SPEC_LIMITS["tensile_strength_mpa"]
    perf_tests.loc[oos_idx, "tensile_strength_mpa"] = np.round(
        rng.uniform(low - 45.0, low - 5.0, size=len(oos_idx)), 3
    )

    manifest: dict[str, object] = {
        "out_of_range_furnace_temp": {
            "runs": [f"R{i:03d}" for i in cfg.INJECTED_OUT_OF_RANGE_RUNS],
            "value": cfg.INJECTED_OUT_OF_RANGE_VALUE,
            "n": len(hit),
        },
        "missing_inspector_id": int(len(miss_idx)),
        "missing_machine_id": int(len(miss_run_idx)),
        "duplicate_inspection_rows": int(cfg.N_INJECTED_DUPLICATE_ROWS),
        "out_of_spec_perf_readings": {
            "column": "tensile_strength_mpa",
            "n": int(cfg.N_INJECTED_OOS_PERF),
            "below": cfg.SPEC_LIMITS["tensile_strength_mpa"][0],
        },
        "constant_column": cfg.CONSTANT_COLUMN_NAME,
        "near_constant_column": {
            "name": cfg.NEAR_CONSTANT_COLUMN_NAME,
            "n_true": int(archived.sum()),
            "share": float(archived.mean()),
        },
        "extra_perf_test_unit": extra["unit_id"].tolist(),
    }
    return runs, inspections, perf_tests, manifest


def generate_all(seed: int = cfg.SEED) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    """生成全部原始表。

    返回 (tables, manifest)：
      tables: material_batches / production_runs / inspection_results / perf_tests
      manifest: 异常注入清单（不落盘为原始数据，但写进指标）
    """
    batches = generate_material_batches(seed)
    runs, latent = generate_production_runs(batches, seed + 1)
    inspections = generate_inspection_results(runs, latent, seed + 2)
    perf = generate_perf_tests(inspections, latent, seed + 3)
    runs, inspections, perf, manifest = inject_known_anomalies(runs, inspections, perf)

    tables = {
        "material_batches": batches,
        "production_runs": runs,
        "inspection_results": inspections,
        "perf_tests": perf,
    }
    # 真值（生成机理里用了哪些因子）单独记录，报告里可与统计结论对照
    manifest["ground_truth_mechanism"] = {
        "primary_factors": ["furnace_temp_c", "material_moisture_pct", "pressure_mpa"],
        "primary_note": "系数最大，5 个种子上均能被识别（见 tests/test_significance.py）",
        "secondary_factors": ["machine_id", "line_speed_mpm", "cooling_rate_cps", "shift"],
        "secondary_note": (
            "机台 M3 为设计的问题机台（较大效应，可检出）；线速/冷却/班次为弱效应，"
            "在 144 个炉次的样本量下不一定检出 —— 这是真实的统计功效限制，不做人为调参掩盖"
        ),
        "indirect_factors": ["material_supplier"],
        "indirect_note": "供应商通过原料含水率间接影响风险，单因子筛选中通常不单独显著",
        "decoy_factors": ["ambient_humidity_pct", "operator_id", "material_particle_size_um"],
        "decoy_note": "真值为零效应，用于检验假阳性控制（BH 校正后不应被稳定判为显著）",
    }
    return tables, manifest
