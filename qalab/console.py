"""控制台输出辅助。

Windows 控制台（以及 CI 上的 Windows runner）默认编码不是 UTF-8，
直接 print 中文会抛 `UnicodeEncodeError: 'charmap' codec can't encode ...`。

这个坑在本项目里踩了三次（CLI、CI 内联脚本、可复现性脚本），
所以统一收敛到这一个函数，任何会打印中文的入口都在开头调用它。
"""

from __future__ import annotations

import sys

__all__ = ["force_utf8_stdout"]


def force_utf8_stdout() -> None:
    """把 stdout / stderr 重配为 UTF-8（失败时静默跳过）。

    对已经重定向到文件或无 `reconfigure` 的流（例如某些测试替身、IDE 控制台）
    保持兼容：`errors="replace"` 保证即使编码不可用也不会因为一个字符而崩溃。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):  # pragma: no cover - 非标准流
            pass
