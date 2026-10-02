"""pytest 公共 fixture。

完整流程（生成数据 + 建模 + 出图）需要十几秒，因此用 session 级 fixture
只跑一次，供多个测试文件复用；单元测试里需要的「小而可控」的数据
由各测试文件自己在内存里构造。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qalab import io as qio  # noqa: E402
from qalab import synth  # noqa: E402


@pytest.fixture(scope="session")
def tables() -> dict:
    """按固定种子生成的四张原始表（session 级，只生成一次）。"""
    data, _manifest = synth.generate_all()
    return data


@pytest.fixture(scope="session")
def manifest() -> dict:
    _data, man = synth.generate_all()
    return man


@pytest.fixture(scope="session")
def wide(tables) -> "object":
    """合并后的分析宽表（未清洗）。"""
    return qio.build_analysis_table(tables)


@pytest.fixture(scope="session")
def clean_wide(wide) -> "object":
    """清洗后的宽表。"""
    clean, _actions = qio.clean_analysis_table(wide)
    return clean


@pytest.fixture(scope="session")
def run_level(clean_wide) -> "object":
    return qio.build_run_level_table(clean_wide)


@pytest.fixture(scope="session")
def batch_level(clean_wide) -> "object":
    return qio.build_batch_level_table(clean_wide)


@pytest.fixture(scope="session")
def modeling(clean_wide):
    """跑一次完整建模（含 ablation），供只读断言复用，避免重复训练拖慢测试。"""
    from qalab import model as qmodel

    return qmodel.run_modeling(clean_wide, with_perf_ablation=True)


@pytest.fixture(scope="session")
def pipeline_result(tmp_path_factory):
    """跑一次完整流程（较少置换次数），供报告/图表/指标类测试复用。"""
    from qalab.pipeline import run_pipeline

    out_dir = tmp_path_factory.mktemp("qalab-reports")
    data_dir = tmp_path_factory.mktemp("qalab-data")
    return run_pipeline(
        out_dir=out_dir,
        data_dir=data_dir,
        make_plots=True,
        n_perm=100,
        log=lambda *a, **k: None,
    )
