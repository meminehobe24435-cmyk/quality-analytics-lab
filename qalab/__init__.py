"""质量分析工作台（Quality Analytics Lab）。

把「生产参数 + 质量检验结果 + 性能测试」三路异构数据合到一起，
做数据质量检查 -> 显著因子识别 -> 统计检验 -> 预测性模型 -> 可视化报告。

⚠️ 本项目使用的全部数据均为**合成**数据（见 qalab/synth.py 的失效机理）。
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
