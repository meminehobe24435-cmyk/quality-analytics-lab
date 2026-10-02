"""全局配置：随机种子、物理规格上下限、列分组、异常注入常量。

本模块是「单一事实来源」：数据生成、质量检查、显著性筛选、建模、报告
全部从这里读取列名与阈值，避免各模块各写一份而互相漂移。
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# 随机性与规模
# --------------------------------------------------------------------------
SEED: int = 20240617
"""全局随机种子。数据生成、切分、交叉验证、模型初始化全部使用它。"""

N_BATCHES: int = 36
"""原料批次数。

原料因子（含水率、粒径、供应商）在**批次级**变化，因此它们的有效样本量是
批次数（36）而不是炉次数（144）或单元数（约 5000）。批次太少会让批次级因子
的检验失去意义（开发中先把批次数设成 12，发现真值零效应的粒径因子在 3/5 个
种子上被误判为显著 —— 因为 12 个批次撑不起一个检验；改成 24 后主因子含水率
仍只在 3/5 个种子上显著，36 个批次才稳定）。
"""

RUNS_PER_BATCH: int = 4
"""每个原料批次的生产炉次数 -> 36 * 4 = 144 个 run。

同一炉次内的单元共享同一个潜在风险 z，因此对「炉次级因子是否显著」而言，
有效样本量是炉次数（144）而不是单元数（约 5000）。
"""

UNITS_PER_RUN_MIN: int = 30
UNITS_PER_RUN_MAX: int = 41
"""每炉次产出的单元数区间（随机抽取）。"""

PERF_TEST_UNIT_FRACTION: float = 0.70
"""抽检做破坏性性能测试的单元比例。"""

PERF_REPEAT_CHOICES: tuple[int, ...] = (1, 1, 1, 1, 2, 3)
"""性能测试复测次数（大多数 1 次，少量复测）-> 制造一对多关系。"""

PERF_MISSING_RATE: float = 0.06
"""性能测试单列的缺失比例（约 6%），用于制造真实的缺失结构。"""

# --------------------------------------------------------------------------
# 物理 / 规格上下限（越界检查依据）
# --------------------------------------------------------------------------
SPEC_LIMITS: dict[str, tuple[float, float]] = {
    "furnace_temp_c": (1420.0, 1580.0),
    "pressure_mpa": (2.20, 3.80),
    "line_speed_mpm": (10.0, 32.0),
    "cooling_rate_cps": (4.0, 12.0),
    "ambient_humidity_pct": (20.0, 90.0),
    "material_moisture_pct": (1.50, 5.50),
    "material_particle_size_um": (20.0, 120.0),
    "tensile_strength_mpa": (300.0, 560.0),
    "hardness_hv": (95.0, 145.0),
    "elongation_pct": (2.50, 18.00),
    "surface_roughness_ra": (0.10, 2.60),
}
"""数值列的量纲/工艺允许范围，越界即数据质量问题（传感器漂移、录入错误）。"""

# --------------------------------------------------------------------------
# 列分组
# --------------------------------------------------------------------------
KEY_COLUMNS: tuple[str, ...] = ("batch_id", "run_id", "unit_id")
LABEL_COL: str = "is_fail"

PRODUCTION_NUMERIC: tuple[str, ...] = (
    "furnace_temp_c",
    "pressure_mpa",
    "line_speed_mpm",
    "cooling_rate_cps",
    "ambient_humidity_pct",
)
MATERIAL_NUMERIC: tuple[str, ...] = (
    "material_moisture_pct",
    "material_particle_size_um",
)
PRODUCTION_CATEGORICAL: tuple[str, ...] = (
    "machine_id",
    "shift",
    "operator_id",
    "material_supplier",
)
PERF_NUMERIC: tuple[str, ...] = (
    "tensile_strength_mpa",
    "hardness_hv",
    "elongation_pct",
    "surface_roughness_ra",
)

MODEL_NUMERIC: tuple[str, ...] = PRODUCTION_NUMERIC + MATERIAL_NUMERIC
MODEL_CATEGORICAL: tuple[str, ...] = PRODUCTION_CATEGORICAL
MODEL_FEATURES: tuple[str, ...] = MODEL_NUMERIC + MODEL_CATEGORICAL
ALL_NUMERIC: tuple[str, ...] = MODEL_NUMERIC + PERF_NUMERIC

LEAKY_COLUMNS: tuple[str, ...] = (
    "inspection_method",
    "inspector_id",
    "defect_type",
)
"""检验环节才产生的字段：标签已确定，纳入特征即信息泄漏，建模时剔除。"""

ID_COLUMNS: tuple[str, ...] = (
    "batch_id",
    "run_id",
    "unit_id",
    "start_time",
    "received_date",
    "inspection_ts",
    "test_ts",
)

# --------------------------------------------------------------------------
# 已知注入的异常（数据生成时写入，质量检查必须能抓到；测试里逐项断言）
# --------------------------------------------------------------------------
N_INJECTED_OUT_OF_RANGE: int = 3
"""被写成 9999.0 的炉温读数（传感器故障）。"""

INJECTED_OUT_OF_RANGE_RUNS: tuple[int, ...] = (7, 23, 41)
INJECTED_OUT_OF_RANGE_VALUE: float = 9999.0

N_INJECTED_DUPLICATE_ROWS: int = 2
"""检验表中完全重复的行数。"""

N_INJECTED_MISSING_INSPECTOR: int = 5
"""检验员缺失的检验记录数。"""

N_INJECTED_MISSING_MACHINE: int = 3
"""机台号缺失的生产炉次数。"""

N_INJECTED_EXTRA_PERF_TESTS: int = 1
"""额外复测到 3 次的单元数（一对多扇出）。"""

N_INJECTED_OOS_PERF: int = 6
"""性能测试中**真实越界**的读数条数（拉伸强度低于规格下限，混在正常抽检里）。"""

CONSTANT_COLUMN_NAME: str = "plant_code"
CONSTANT_COLUMN_VALUE: str = "PLANT-SH-01"
NEAR_CONSTANT_COLUMN_NAME: str = "is_archived"
NEAR_CONSTANT_TRUE_RATE: float = 0.004

# --------------------------------------------------------------------------
# 统计与建模参数
# --------------------------------------------------------------------------
ALPHA: float = 0.05
"""显著性水平（Benjamini-Hochberg 的 FDR 控制目标）。"""

CV_FOLDS: int = 5
TEST_SIZE: float = 0.20
VAL_SIZE: float = 0.20
"""先切 20% 测试集，再从剩余切 20% 作验证集（用于选阈值）。"""

DRIFT_BINS: int = 10
"""PSI 分箱数。"""

N_PERM: int = 1000
"""簇（炉次）置换检验的置换次数。固定次数 + 固定种子 = 可复现。"""

DRIFT_KS_ALPHA: float = 0.01
"""漂移判定：KS 检验的 p 值门槛。"""

DRIFT_KS_STAT_THRESHOLD: float = 0.20
"""漂移判定：KS 统计量的**工程效应量**门槛。

KS 统计量 = 两条经验分布曲线的最大垂直距离。0.20 表示「至少 20% 的样本
在分布上的位置发生了移动」，这是工程上能感觉到的差异。
"""

DRIFT_PSI_THRESHOLD: float = 0.25
"""漂移判定：PSI 门槛（PSI>0.25 为业界公认的「显著漂移」档）。

⚠️ 开发中实测踩坑：一开始用「KS 的 p 值 + PSI≥0.1」，结果 11 个数值列里有
9 个被判成漂移，等于没有信息量。两个原因：
  (1) n≈5000 时 KS 对任何微小差异都显著 —— 统计显著 ≠ 工程重要；
  (2) PSI 在 n=100 量级的零假设期望值本身就有 0.08~0.1，
      把阈值定在 0.1 会导致大量假报警。
改成「KS p<0.01 且 KS 统计量≥0.20 且 PSI≥0.25」三重门槛后，
只有真正由原料批次更换造成的 2 个列被判为漂移。
"""

NEAR_CONSTANT_SHARE: float = 0.99
"""某一取值占比超过该阈值的列视为近常量列。"""

DRIFT_BASELINE_FRACTION: float = 0.25
"""漂移基线 = 按 start_time 排序后**最早 25% 的炉次**（约 36 个炉次）。

⚠️ 开发中实测踩坑：最初把「单个原料批次」当基线，结果基线只有 4 个炉次，
PSI 分箱极不稳定，11 个数值列里 9 个被判成「漂移」，等于没有信息量。
基线的独立观测数必须够多，漂移判据才有意义。
"""

DRIFT_BASELINE_BATCH: str = "B01"
"""备用基线（仅在缺少 start_time 时按批次取基线）。"""

# --------------------------------------------------------------------------
# 路径
# --------------------------------------------------------------------------
PACKAGE_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = PACKAGE_DIR.parent
DEFAULT_DATA_DIR: Path = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUT_DIR: Path = PROJECT_ROOT / "reports"

RAW_FILES: dict[str, str] = {
    "material_batches": "material_batches.csv",
    "production_runs": "production_runs.csv",
    "inspection_results": "inspection_results.csv",
    "perf_tests": "perf_tests.csv",
}
"""三路异构数据 + 原料主数据，落盘为 4 个 CSV。"""
