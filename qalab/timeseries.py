"""时间序列 / 趋势与漂移分析（**按批次顺序的序列分析，不是高频时序**）。

⚠️ 边界先说清楚：本项目的数据是「36 个原料批次、每批 4 个炉次」的**生产顺序序列**，
不是秒级/分钟级的高频时序。因此这里做的是：

  1. **移动平均**：把批次级指标的短期波动压掉，看清趋势；
  2. **趋势判断**（两种口径互相印证）：
     - 线性趋势：最小二乘拟合斜率 + 斜率显著性（scipy.stats.linregress）；
     - **Mann-Kendall 趋势检验**（自己实现，含并列值校正）+ Sen's 斜率：
       非参数、不假定正态、对离群值稳健，是环境/质量领域做趋势检验的标准做法；
  3. **最朴素的预测基线**：简单指数平滑（SES，平滑系数由一步预测误差最小化选出）
     与 AR(1)（用滞后一阶回归拟合），预测**下一个批次**的指标；
     误差用**扩张窗口的一步预测回测**衡量（在每个时间点只用它之前的数据拟合），
     并与「用历史均值预测」「用上一个值预测」两个更笨的基线对比；
  4. 附上 **Ljung-Box 风格的自相关检查**：报告滞后 1 阶自相关系数及其显著性，
     判断这个序列到底有没有可利用的记忆。

**结论怎么读**：如果趋势检验不显著、自相关也不显著，正确的结论是
「过程处于统计受控状态，没有"最近变差了"的证据」—— 这是有价值的阴性结论，
而不是失败。此时改善方向应回到已识别的显著因子（炉温/压力/机台/原料含水率）。
检测器本身的功效由单元测试保证（注入已知趋势必须被检出）。

退化输入（全部有测试）：只有 1 个批次、序列全相等（无方差）、点数不足、
含缺失值、空序列。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import config as cfg

__all__ = [
    "moving_average",
    "linear_trend_test",
    "mann_kendall_test",
    "sen_slope",
    "simple_exponential_smoothing",
    "fit_ar1",
    "lag1_autocorrelation",
    "expanding_window_backtest",
    "batch_series",
    "run_series",
    "run_timeseries",
]


# --------------------------------------------------------------------------
# 序列构造
# --------------------------------------------------------------------------
def batch_series(df: pd.DataFrame, value_col: str, label: str = cfg.LABEL_COL) -> pd.DataFrame:
    """按**批次顺序**聚合出一条序列。

    批次顺序取自各批次内最早的 start_time（而不是批号字符串排序），
    这样即使批号命名规则变了，序列顺序仍然是生产顺序。

    `value_col` 为 `label` 时聚合不合格率（fail_rate），否则聚合该列的均值。
    """
    if "batch_id" not in df.columns:
        return pd.DataFrame(columns=["batch_id", "value", "n_units", "order_time"])
    g = df.groupby("batch_id", as_index=False)
    agg = g.agg(
        value=(value_col, "mean"),
        n_units=(label, "size"),
    )
    if "start_time" in df.columns:
        times = g["start_time"].min().rename(columns={"start_time": "order_time"})
        agg = agg.merge(times, on="batch_id", how="left", validate="1:1")
        agg = agg.sort_values("order_time", kind="stable")
    else:
        agg["order_time"] = pd.NaT
        agg = agg.sort_values("batch_id", kind="stable")
    return agg.reset_index(drop=True)


def run_series(df: pd.DataFrame, value_col: str, label: str = cfg.LABEL_COL) -> pd.DataFrame:
    """按炉次顺序（生产顺序）聚合出的序列，用作更细粒度的旁证。"""
    if "run_id" not in df.columns:
        return pd.DataFrame(columns=["run_id", "value", "n_units", "order_time"])
    g = df.groupby("run_id", as_index=False)
    agg = g.agg(value=(value_col, "mean"), n_units=(label, "size"))
    if "start_time" in df.columns:
        times = g["start_time"].min().rename(columns={"start_time": "order_time"})
        agg = agg.merge(times, on="run_id", how="left", validate="1:1")
        agg = agg.sort_values("order_time", kind="stable")
    else:
        agg["order_time"] = pd.NaT
        agg = agg.sort_values("run_id", kind="stable")
    return agg.reset_index(drop=True)


# --------------------------------------------------------------------------
# 移动平均
# --------------------------------------------------------------------------
def moving_average(values, window: int = 5) -> pd.Series:
    """移动平均（居中不做，保持因果性：第 t 个值只用 t-window+1..t）。

    窗口小于 2 或序列太短时，返回全 NaN 的序列并保留原长度（调用方据此跳过绘图）。
    """
    s = pd.Series(pd.to_numeric(pd.Series(values), errors="coerce"), dtype="float64")
    if window < 2 or len(s) < window:
        return pd.Series(np.nan, index=s.index, dtype="float64")
    return s.rolling(window=window, min_periods=window).mean()


# --------------------------------------------------------------------------
# 趋势检验
# --------------------------------------------------------------------------
def _clean_series(values) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def linear_trend_test(values) -> dict[str, object]:
    """线性趋势：最小二乘斜率 + 双侧 p 值 + R²。

    点数 < 3 或自变量无方差时返回 None 并说明（斜率无定义）。
    """
    y = _clean_series(values)
    out: dict[str, object] = {
        "test": "linear_trend_ols",
        "n": int(len(y)),
        "slope": None,
        "intercept": None,
        "p_value": None,
        "r2": None,
        "slope_per_100_steps": None,
        "trend": "无法判定",
        "note": "",
    }
    if len(y) < 3:
        out["note"] = "点数 < 3，线性趋势无定义"
        return out
    x = np.arange(len(y), dtype=float)
    if np.std(x) == 0:
        out["note"] = "自变量无方差，斜率无定义"
        return out
    res = sps.linregress(x, y)
    out["slope"] = float(res.slope)
    out["intercept"] = float(res.intercept)
    out["p_value"] = float(res.pvalue)
    out["r2"] = float(res.rvalue**2)
    out["slope_per_100_steps"] = float(res.slope * 100)
    out["trend"] = _trend_label(res.pvalue, res.slope)
    return out


def _trend_label(p_value: float | None, slope: float | None, alpha: float = cfg.ALPHA) -> str:
    if p_value is None or slope is None:
        return "无法判定"
    if p_value >= alpha:
        return "无显著趋势"
    return "显著上升趋势" if slope > 0 else "显著下降趋势"


def sen_slope(values) -> float | None:
    """Sen's 斜率：所有点对斜率的中位数（对离群值稳健的非参数趋势量）。"""
    y = _clean_series(values)
    n = len(y)
    if n < 2:
        return None
    slopes = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            slopes.append((y[j] - y[i]) / (j - i))
    if not slopes:
        return None
    return float(np.median(slopes))


def mann_kendall_test(values, alpha: float = cfg.ALPHA) -> dict[str, object]:
    """Mann-Kendall 趋势检验（自己实现，含并列值校正）+ Sen's 斜率。

    统计量：
        S = Σ_{i<j} sign(x_j - x_i)
        Var(S) = [n(n-1)(2n+5) - Σ_t t(t-1)(2t+5)] / 18      （t 为并列组的长度）
        Z = (S-1)/√Var(S) （S>0）、(S+1)/√Var(S) （S<0）、0 （S=0）
        p = 2 * (1 - Φ(|Z|))
    序列全相等（Var(S)=0）时 Z 与 p 都无定义，返回 None 而不是硬给一个 0/1。
    """
    y = _clean_series(values)
    n = len(y)
    out: dict[str, object] = {
        "test": "mann_kendall",
        "n": int(n),
        "S": None,
        "var_s": None,
        "Z": None,
        "p_value": None,
        "sen_slope": None,
        "trend": "无法判定",
        "note": "",
    }
    if n < 3:
        out["note"] = "点数 < 3，趋势检验无定义"
        return out

    s = 0
    for i in range(n - 1):
        s += int(np.sum(np.sign(y[i + 1 :] - y[i])))
    out["S"] = int(s)

    # 并列值校正：把每个取值出现的次数 t 代入
    _, counts = np.unique(y, return_counts=True)
    tie_term = float(np.sum(counts * (counts - 1) * (2 * counts + 5)))
    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0
    out["var_s"] = float(var_s)
    if var_s <= 0:
        out["note"] = "序列方差为 0（全部取值相同），Z 与 p 无定义"
        out["sen_slope"] = sen_slope(y)
        return out

    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0
    p = 2.0 * (1.0 - sps.norm.cdf(abs(z)))
    out["Z"] = float(z)
    out["p_value"] = float(p)
    out["sen_slope"] = sen_slope(y)
    out["trend"] = _trend_label(p, out["sen_slope"], alpha)
    if out["trend"] == "无显著趋势":
        out["note"] = "未检出显著单调趋势（在 alpha=%s 下）" % alpha
    return out


# --------------------------------------------------------------------------
# 自相关
# --------------------------------------------------------------------------
def lag1_autocorrelation(values, alpha: float = cfg.ALPHA) -> dict[str, object]:
    """滞后 1 阶自相关（以及它的显著性）。

    用途：判断序列有没有"记忆"。若无自相关，则任何预测器都不可能比"用均值预测"
    好多少 —— 这本身就是结论，能挡住"最近好像变差了"这种错觉。
    """
    y = _clean_series(values)
    out: dict[str, object] = {
        "n": int(len(y)),
        "r1": None,
        "p_value": None,
        "significant": False,
        "note": "",
    }
    if len(y) < 4:
        out["note"] = "点数 < 4，自相关无定义"
        return out
    if np.std(y) == 0:
        out["note"] = "序列无方差，自相关无定义"
        return out
    a, b = y[:-1], y[1:]
    if np.std(a) == 0 or np.std(b) == 0:
        out["note"] = "滞后段无方差，自相关无定义"
        return out
    res = sps.pearsonr(a, b)
    out["r1"] = float(res.statistic)
    out["p_value"] = float(res.pvalue)
    out["significant"] = bool(res.pvalue < alpha)
    return out


# --------------------------------------------------------------------------
# 预测基线
# --------------------------------------------------------------------------
def simple_exponential_smoothing(values, alpha: float | None = None) -> dict[str, object]:
    """简单指数平滑（SES）。

    `alpha` 未指定时，在 (0, 1) 上按**一步预测误差平方和**最小化来选（网格搜索，
    固定网格保证可复现）。返回平滑后的序列、选中的 alpha、以及下一个点的预测值
    （预测值 = 平滑序列的最后一个水平）。
    """
    y = _clean_series(values)
    out: dict[str, object] = {
        "method": "simple_exponential_smoothing",
        "n": int(len(y)),
        "alpha": None,
        "fitted": [],
        "next_forecast": None,
        "note": "",
    }
    if len(y) < 3:
        out["note"] = "点数 < 3，指数平滑无定义"
        return out
    if np.std(y) == 0:
        # 全相等：任何 alpha 都给出同一个常数预测
        out["alpha"] = 0.5
        out["fitted"] = [float(y[0])] * len(y)
        out["next_forecast"] = float(y[0])
        out["note"] = "序列无方差，预测值等于该常数"
        return out

    grid = np.round(np.arange(0.05, 1.0, 0.05), 4)
    if alpha is not None:
        grid = np.array([float(alpha)])

    def run(a: float) -> tuple[np.ndarray, float]:
        level = y[0]
        fitted = np.empty(len(y), dtype=float)
        fitted[0] = level
        sse = 0.0
        for t in range(1, len(y)):
            fitted[t] = level
            sse += (y[t] - level) ** 2
            level = a * y[t] + (1 - a) * level
        return fitted, sse

    scores = [(run(float(a))[1], float(a)) for a in grid]
    best_sse, best_alpha = min(scores, key=lambda t: (t[0], t[1]))
    fitted, _ = run(best_alpha)
    out["alpha"] = best_alpha
    out["sse"] = float(best_sse)
    out["fitted"] = [float(v) for v in fitted]
    out["next_forecast"] = float(fitted[-1])
    return out


def fit_ar1(values) -> dict[str, object]:
    """AR(1)：x_t = c + φ·x_{t-1} + ε，用最小二乘拟合，φ 也报显著性。

    φ 无定义（滞后段无方差）或点数不足时返回 note，不硬给结果。
    """
    y = _clean_series(values)
    out: dict[str, object] = {
        "method": "ar1",
        "n": int(len(y)),
        "phi": None,
        "intercept": None,
        "p_value_phi": None,
        "next_forecast": None,
        "note": "",
    }
    if len(y) < 5:
        out["note"] = "点数 < 5，AR(1) 无定义"
        return out
    x_lag, x_now = y[:-1], y[1:]
    if np.std(x_lag) == 0:
        out["note"] = "滞后段无方差，AR(1) 系数无定义"
        return out
    res = sps.linregress(x_lag, x_now)
    out["phi"] = float(res.slope)
    out["intercept"] = float(res.intercept)
    out["p_value_phi"] = float(res.pvalue)
    out["r2"] = float(res.rvalue**2)
    out["next_forecast"] = float(res.intercept + res.slope * y[-1])
    return out


def expanding_window_backtest(values, min_train: int = 5) -> dict[str, object]:
    """扩张窗口的一步预测回测：每个 t 只用 t 之前的数据拟合，再预测 t。

    报告三种预测器的一步 MAE/RMSE：
      - `ses`：简单指数平滑（alpha 每次只用历史选）
      - `ar1`：AR(1)
      - `mean`：历史均值（最笨但常常很强的基线）
      - `last`：上一个观测值（naive 基线）
    训练点不足 `min_train` 时返回 note。
    """
    y = _clean_series(values)
    out: dict[str, object] = {
        "n": int(len(y)),
        "n_backtest_points": 0,
        "min_train": int(min_train),
        "predictors": {},
        "note": "",
    }
    if len(y) < min_train + 2:
        out["note"] = f"点数不足（需要至少 {min_train + 2} 个点做回测）"
        return out

    preds: dict[str, list[float]] = {"ses": [], "ar1": [], "mean": [], "last": []}
    actual: list[float] = []
    for t in range(min_train, len(y)):
        hist = y[:t]
        actual.append(float(y[t]))
        ses = simple_exponential_smoothing(hist)
        preds["ses"].append(
            float(ses["next_forecast"]) if ses["next_forecast"] is not None else float(np.mean(hist))
        )
        ar = fit_ar1(hist)
        preds["ar1"].append(
            float(ar["next_forecast"]) if ar["next_forecast"] is not None else float(np.mean(hist))
        )
        preds["mean"].append(float(np.mean(hist)))
        preds["last"].append(float(hist[-1]))

    actual_arr = np.asarray(actual, dtype=float)
    out["n_backtest_points"] = int(len(actual_arr))
    for name, values_pred in preds.items():
        p = np.asarray(values_pred, dtype=float)
        out["predictors"][name] = {
            "mae": float(np.mean(np.abs(actual_arr - p))),
            "rmse": float(np.sqrt(np.mean((actual_arr - p) ** 2))),
        }
    best = min(out["predictors"], key=lambda n: out["predictors"][n]["mae"])
    out["best_predictor_by_mae"] = best
    out["note"] = (
        f"一步预测回测 {out['n_backtest_points']} 个点；"
        f"MAE 最小的是 {best}（{out['predictors'][best]['mae']:.4f}）"
    )
    return out


# --------------------------------------------------------------------------
# 组装
# --------------------------------------------------------------------------
def _analyze_one_series(
    series: pd.DataFrame, name: str, window: int, alpha: float
) -> dict[str, object]:
    values = series["value"] if "value" in series.columns else pd.Series([], dtype=float)
    ma = moving_average(values, window=window)
    result: dict[str, object] = {
        "name": name,
        "n_points": int(len(series)),
        "window": int(window),
        "values": [None if pd.isna(v) else float(v) for v in pd.to_numeric(values, errors="coerce")],
        "moving_average": [None if pd.isna(v) else float(v) for v in ma],
        "linear_trend": linear_trend_test(values),
        "mann_kendall": mann_kendall_test(values, alpha=alpha),
        "autocorrelation_lag1": lag1_autocorrelation(values, alpha=alpha),
        "exponential_smoothing": simple_exponential_smoothing(values),
        "ar1": fit_ar1(values),
        "backtest": expanding_window_backtest(values),
        "note": "",
    }
    mk = result["mann_kendall"]
    ac = result["autocorrelation_lag1"]
    if mk["p_value"] is None:
        result["note"] = f"趋势无法判定：{mk['note']}"
    elif mk["trend"] == "无显著趋势" and not ac["significant"]:
        result["note"] = (
            "无显著趋势且无显著自相关 -> 该序列处于统计受控状态，"
            "不存在「最近变差」的统计证据；改善方向应回到已识别的显著因子"
        )
    elif mk["trend"] != "无显著趋势" and not ac["significant"]:
        result["note"] = "检出显著趋势但无可利用的自相关（可直接用趋势外推做粗预测）"
    elif mk["trend"] == "无显著趋势" and ac["significant"]:
        result["note"] = "无趋势但有显著自相关（短期波动有记忆，可用 AR(1) 做一步预测）"
    else:
        result["note"] = "既有趋势又有自相关（需用带趋势的时序模型）"
    return result


def run_timeseries(
    df: pd.DataFrame,
    label: str = cfg.LABEL_COL,
    value_cols: tuple[str, ...] | None = None,
    window: int = 5,
    alpha: float = cfg.ALPHA,
) -> tuple[dict[str, object], dict[str, object]]:
    """按批次顺序跑趋势与漂移分析 + 下一批预测。

    分析对象：批次不合格率（主）+ 若干连续质量指标（批次均值）+ 炉次级不合格率（旁证）。
    返回 (可 JSON 化的结果, 绘图数据)。
    """
    result: dict[str, object] = {
        "scope_note": (
            "这是**按生产顺序（批次/炉次）排列的序列分析**，不是高频时序："
            "数据是 36 个原料批次、每批 4 个炉次，没有秒/分钟级采样，"
            "因此不做季节性分解，也不声称能做短期实时预测"
        ),
        "n_batches": 0,
        "n_runs": 0,
        "window": int(window),
        "series": {},
        "primary_series": "batch_fail_rate",
        "conclusion": "",
    }
    extras: dict[str, object] = {}

    if "batch_id" not in df.columns or df.empty:
        result["conclusion"] = "缺少 batch_id 或数据为空，无法做序列分析"
        return result, extras

    main = batch_series(df, label, label=label)
    result["n_batches"] = int(len(main))
    result["n_runs"] = int(df["run_id"].nunique()) if "run_id" in df.columns else 0

    series_map: dict[str, pd.DataFrame] = {"batch_fail_rate": main}
    for col in value_cols or ():
        if col in df.columns:
            series_map[f"batch_mean_{col}"] = batch_series(df, col, label=label)
    if "run_id" in df.columns:
        series_map["run_fail_rate"] = run_series(df, label, label=label)

    if len(main) < 2:
        result["conclusion"] = (
            f"只有 {len(main)} 个批次，序列太短：趋势检验与预测都无法给出有意义结论"
        )
        for name, s in series_map.items():
            result["series"][name] = _analyze_one_series(s, name, window, alpha)
        extras["series"] = series_map
        return result, extras

    for name, s in series_map.items():
        result["series"][name] = _analyze_one_series(s, name, window, alpha)

    primary = result["series"]["batch_fail_rate"]
    mk = primary["mann_kendall"]
    bt = primary["backtest"]
    forecast_bits = []
    for method in ("exponential_smoothing", "ar1"):
        fc = primary[method].get("next_forecast")
        if fc is not None:
            forecast_bits.append(f"{method}={fc:.4f}")
    result["next_batch_forecast"] = {
        "exponential_smoothing": primary["exponential_smoothing"].get("next_forecast"),
        "ar1": primary["ar1"].get("next_forecast"),
        "historical_mean": float(np.mean(pd.to_numeric(main["value"], errors="coerce"))),
        "last_observed": float(pd.to_numeric(main["value"], errors="coerce").iloc[-1]),
    }
    result["forecast_one_step_error"] = bt.get("predictors", {})
    result["trend_summary"] = {
        "mann_kendall_p": mk.get("p_value"),
        "mann_kendall_trend": mk.get("trend"),
        "sen_slope": mk.get("sen_slope"),
        "linear_p": primary["linear_trend"].get("p_value"),
        "linear_slope": primary["linear_trend"].get("slope"),
    }
    result["conclusion"] = (
        f"批次不合格率序列（{len(main)} 个点）：Mann-Kendall {mk['trend']}"
        f"（p={mk['p_value'] if mk['p_value'] is None else round(mk['p_value'], 4)}）；"
        f"下一个批次的预测 {('、'.join(forecast_bits)) or '无法给出'}；"
        f"一步预测误差见 forecast_one_step_error。{primary['note']}"
    )
    extras["series"] = series_map
    extras["primary"] = main
    return result, extras
