"""复现性验证：连跑两次完整流程，逐字节比对产物。

用法: py -3.12 tools/check_reproducibility.py [--n-perm 200]

退出码 0 表示 metrics.json / 实验报告.md / CSV / PNG 全部逐字节一致。
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qalab.console import force_utf8_stdout  # noqa: E402
from qalab.pipeline import run_pipeline  # noqa: E402

# Windows 控制台不是 UTF-8，下面要打印中文表头与「一致/不一致」，必须先重配
force_utf8_stdout()

TRACKED = [
    "metrics.json",
    "实验报告.md",
    "factor_ranking.csv",
    "model_comparison.csv",
    "significance_tests.csv",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(out_dir: Path) -> dict[str, str]:
    files = [out_dir / name for name in TRACKED]
    files += sorted(out_dir.glob("*.png"))
    return {p.name: sha256(p) for p in files if p.exists()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-perm", type=int, default=200, help="置换次数（验证用较少次数即可）")
    args = parser.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="qalab-repro-"))
    data_dir = tmp / "data"
    try:
        out1, out2 = tmp / "run1", tmp / "run2"
        print("第一次运行 ...")
        run_pipeline(out_dir=out1, data_dir=data_dir, regenerate_data=True, n_perm=args.n_perm, log=lambda *a: None)
        print("第二次运行 ...")
        run_pipeline(out_dir=out2, data_dir=data_dir, n_perm=args.n_perm, log=lambda *a: None)

        snap1, snap2 = snapshot(out1), snapshot(out2)
        names = sorted(set(snap1) | set(snap2))
        print(f"\n{'文件':32s} {'一致':6s} sha256(前 16 位)")
        print("-" * 72)
        all_ok = True
        for name in names:
            h1, h2 = snap1.get(name), snap2.get(name)
            ok = h1 is not None and h1 == h2
            all_ok &= ok
            print(f"{name:32s} {'一致' if ok else '不一致':6s} {(h1 or '缺失')[:16]}")

        metrics1, metrics2 = snap1.get("metrics.json"), snap2.get("metrics.json")
        print()
        print(f"metrics.json       sha256 = {metrics1}")
        print(f"实验报告.md        sha256 = {snap1.get('实验报告.md')}")
        print(f"PNG 数量           {sum(1 for n in names if n.endswith('.png'))}")

        # run_meta.json 应当**不同**（含耗时与时间戳）
        meta_differs = sha256(out1 / "run_meta.json") != sha256(out2 / "run_meta.json")
        print(f"run_meta.json 两次不同（预期如此，含耗时）: {meta_differs}")

        print()
        if all_ok:
            print("结论：除 run_meta.json 外，全部产物逐字节一致 —— 可复现性成立。")
            return 0
        print("结论：存在不一致的产物，可复现性不成立。")
        return 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
