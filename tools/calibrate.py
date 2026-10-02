"""开发期校准脚本：检查合成数据的信号是否按设计可被识别（不入库为正式代码）。

用法: py -3.12 tools/calibrate.py [n_seeds]
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from qalab import config as cfg
from qalab import io, significance, synth
from qalab.console import force_utf8_stdout

force_utf8_stdout()

PRIMARY = ["furnace_temp_c", "material_moisture_pct", "pressure_mpa"]
SECONDARY = ["line_speed_mpm", "cooling_rate_cps", "machine_id", "shift", "material_supplier"]
DECOY = ["ambient_humidity_pct", "operator_id", "material_particle_size_um"]


def one_seed(seed: int) -> pd.DataFrame:
    tables, _ = synth.generate_all(seed)
    wide = io.build_analysis_table(tables)
    clean, _ = io.clean_analysis_table(wide)
    run_df = io.build_run_level_table(clean)
    batch_df = io.build_batch_level_table(clean)
    tests = significance.screen_factors(clean, run_df, batch_df, n_perm=300, seed=seed)
    rank = significance.rank_factors(tests)
    rank["group"] = np.where(
        rank["factor"].isin(PRIMARY), "P", np.where(rank["factor"].isin(SECONDARY), "S", "D")
    )
    rank["seed"] = seed
    return rank


def main(n_seeds: int = 5) -> None:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    frames = [one_seed(cfg.SEED + i * 1000) for i in range(n_seeds)]
    allr = pd.concat(frames, ignore_index=True)

    cols = [
        "factor",
        "group",
        "factor_level",
        "n_independent",
        "unit_naive_q_value",
        "unit_q_value",
        "factor_q_value",
        "unit_effect_size",
        "mutual_info",
    ]
    print(f"=== seed={cfg.SEED} 的排序（factor_q_value = 因子自身层级权威口径） ===")
    print(frames[0][cols].to_string(index=False))

    print("\n=== 跨种子稳定性（显著次数 / 种子数）===")
    summary = (
        allr.groupby(["group", "factor"])[
            ["unit_naive_significant", "unit_significant", "factor_significant"]
        ]
        .sum()
        .astype(int)
        .reset_index()
    )
    summary.columns = ["group", "factor", "朴素单元级显著", "簇置换显著", "因子层级显著"]
    print(summary.to_string(index=False))

    print("\n=== fail rate / 规模（首种子）===")
    tables, _ = synth.generate_all(cfg.SEED)
    insp = tables["inspection_results"]
    print(
        "units=%d  fails=%d  rate=%.4f  runs=%d  perf_rows=%d"
        % (
            len(insp),
            int(insp["is_fail"].sum()),
            insp["is_fail"].mean(),
            tables["production_runs"].shape[0],
            tables["perf_tests"].shape[0],
        )
    )
    perf = tables["perf_tests"]
    print("\nperf describe (raw, 含越界检查):")
    print(perf[list(cfg.PERF_NUMERIC)].describe().loc[["mean", "std", "min", "max"]].round(3).to_string())
    wide = io.build_analysis_table(tables)
    from qalab import quality

    oob = quality.out_of_range_report(wide)
    print("\n越界统计:")
    print(oob[oob["n_out_of_range"] > 0][["column", "n_below", "n_above", "n_out_of_range", "first_example"]].to_string(index=False))


if __name__ == "__main__":
    import sys

    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
