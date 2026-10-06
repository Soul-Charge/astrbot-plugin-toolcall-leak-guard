"""把模型「写成纯文本的工具调用」还原成结构化调用。

背景
----
上游 endpoint 偶尔不返回原生 tool_calls，而是把调用写成一段文本塞进
message.content：标签由 calls / invoke / parameter 三个词组成，每个标签前还
带着模型自己的特殊标记（两个全角竖线 + DSML + 两个全角竖线）。AstrBot 只认识
原生 tool_calls，于是这段文本被当成回复原文直接发进了群。

本模块只做「纯文本 -> 结构化调用」的转换，不依赖 AstrBot，方便单独单测。

设计原则
--------
1. 先保证「不漏」：只要文本里出现这类标签块，就一定从要发出的正文里摘掉。
2. 再谈「还原」：只有工具名确实在本次请求的工具集里、且块外没有别的正文时，
   才把它变成真正的工具调用去执行。
3. 解析失败（被截断、标签变形）时不抛异常，返回 unparsed=True，交给上层只做摘除。
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "DSML_MARKER",
    "LeakedToolCall",
    "LeakParseResult",
    "extract_leaked_tool_calls",
    "looks_like_leak",
]

# ---------------------------------------------------------------------------
# 标签形状
# ---------------------------------------------------------------------------
_BAR = "\uff5c"  # 全角竖线
DSML_MARKER = _BAR * 2 + "DSML" + _BAR * 2

_TAG_OPEN = r"<\s*(?:" + re.escape(DSML_MARKER) + r"\s*)?"
_TAG_CLOSE = r"</\s*(?:" + re.escape(DSML_MARKER) + r"\s*)?"

_BLOCK = "calls"
_INVOKE = "invoke"
_PARAM = "parameter"

_RE_BLOCK = re.compile(
    _TAG_OPEN + _BLOCK + r"\s*>(?P<body>.*?)" + _TAG_CLOSE + _BLOCK + r"\s*>",
    re.DOTALL,
)
_RE_INVOKE = re.compile(
    _TAG_OPEN
    + _INVOKE
    + r"\s*(?P<attrs>[^>]*)>(?P<body>.*?)"
    + _TAG_CLOSE
    + _INVOKE
    + r"\s*>",
    re.DOTALL,
)
_RE_PARAM = re.compile(
    _TAG_OPEN
    + _PARAM
    + r"\s*(?P<attrs>[^>]*)>(?P<body>.*?)"
    + _TAG_CLOSE
    + _PARAM
    + r"\s*>",
    re.DOTALL,
)
# 截断兜底：参数标签没闭合时，切到下一个开标签或结尾
_RE_PARAM_LOOSE = re.compile(
    _TAG_OPEN
    + _PARAM
    + r"\s*(?P<attrs>[^>]*)>(?P<body>.*?)(?="
    + _TAG_OPEN
    + r"(?:"
    + _BLOCK
    + r"|"
    + _INVOKE
    + r"|"
    + _PARAM
    + r")\b|$)",
    re.DOTALL,
)
_RE_ANY_OPEN = re.compile(
    _TAG_OPEN + r"(?:" + _BLOCK + r"|" + _INVOKE + r"|" + _PARAM + r")\b"
)
_RE_ANY_PARAM_OPEN = re.compile(_TAG_OPEN + _PARAM + r"\b")

_ATTR_CACHE: dict[str, re.Pattern[str]] = {}
_NOT_JSON = object()


@dataclass
class LeakedToolCall:
    """一个还原出来的工具调用。"""

    name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class LeakParseResult:
    """解析结果。

    Attributes:
        found: 文本里确实出现了这类标签块。
        calls: 成功还原出来的调用（可能为空，例如被截断）。
        leftover: 摘掉标签块之后剩下的正文。
        unparsed: 出现了标签块，但一个调用都没解出来。
    """

    found: bool = False
    calls: list[LeakedToolCall] = field(default_factory=list)
    leftover: str = ""
    unparsed: bool = False


def looks_like_leak(text: str) -> bool:
    """快速判断文本里是否可能出现这类标签块（便宜的前置过滤）。"""
    if not text:
        return False
    if DSML_MARKER in text:
        return True
    return _RE_ANY_OPEN.search(text) is not None


def extract_leaked_tool_calls(text: str) -> LeakParseResult:
    """从文本里摘出泄漏的工具调用。"""
    result = LeakParseResult(leftover=text or "")
    if not text or not looks_like_leak(text):
        return result

    spans: list[tuple[int, int]] = []
    bodies: list[str] = []

    for match in _RE_BLOCK.finditer(text):
        spans.append((match.start(), match.end()))
        bodies.append(match.group("body"))

    if not spans:
        # 没有外层块，可能是裸的 invoke
        for match in _RE_INVOKE.finditer(text):
            spans.append((match.start(), match.end()))
            bodies.append(match.group(0))

    if not spans:
        # 出现了标签但整段没闭合（被截断）：从第一个开标签开始切掉
        first = _RE_ANY_OPEN.search(text)
        if first is None:
            return result
        result.found = True
        result.unparsed = True
        result.leftover = text[: first.start()]
        return result

    result.found = True
    calls: list[LeakedToolCall] = []
    for body in bodies:
        calls.extend(_parse_invokes(body))
    result.calls = calls
    if not calls:
        result.unparsed = True
    result.leftover = _cut_spans(text, spans)
    return result


def _parse_invokes(body: str) -> list[LeakedToolCall]:
    calls: list[LeakedToolCall] = []
    for match in _RE_INVOKE.finditer(body):
        name = _attr(match.group("attrs"), "name")
        if not name:
            continue
        inner = match.group("body")
        args: dict[str, Any] = {}
        for param in _RE_PARAM.finditer(inner):
            key = _attr(param.group("attrs"), "name")
            if not key:
                continue
            args[key] = _convert_value(
                param.group("body"),
                _attr(param.group("attrs"), "string"),
            )
        if not args and _RE_ANY_PARAM_OPEN.search(inner):
            # 参数标签没闭合时的兜底
            for param in _RE_PARAM_LOOSE.finditer(inner):
                key = _attr(param.group("attrs"), "name")
                if not key:
                    continue
                args[key] = _convert_value(
                    param.group("body"),
                    _attr(param.group("attrs"), "string"),
                )
        calls.append(LeakedToolCall(name=name.strip(), args=args))
    return calls


def _attr(attrs: str, key: str) -> str | None:
    pattern = _ATTR_CACHE.get(key)
    if pattern is None:
        pattern = re.compile(
            r"\b" + re.escape(key) + r"\s*=\s*(?:\"([^\"]*)\"|'([^']*)')"
        )
        _ATTR_CACHE[key] = pattern
    match = pattern.search(attrs or "")
    if match is None:
        return None
    return match.group(1) if match.group(1) is not None else match.group(2)


def _try_json(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return _NOT_JSON


def _convert_value(raw: str, string_flag: str | None) -> Any:
    """按 parameter 的 string 属性决定取值类型。

    string="true"  -> 原始字符串（做 XML 实体反转义）
    string="false" -> JSON（数组 / 数字 / 布尔 / 对象）
    没有该属性      -> 像 JSON 就按 JSON，否则当字符串
    """
    text = html.unescape(raw or "")
    flag = (string_flag or "").strip().lower()
    if flag == "true":
        return text.strip()
    if flag == "false":
        parsed = _try_json(text)
        return text.strip() if parsed is _NOT_JSON else parsed

    stripped = text.strip()
    if stripped[:1] in ("{", "["):
        parsed = _try_json(stripped)
        if parsed is not _NOT_JSON:
            return parsed
    if stripped in ("true", "false", "null") or re.fullmatch(
        r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", stripped
    ):
        parsed = _try_json(stripped)
        if parsed is not _NOT_JSON:
            return parsed
    return stripped


def _cut_spans(text: str, spans: list[tuple[int, int]]) -> str:
    """把所有命中区间从文本里摘掉（自动合并重叠区间）。"""
    if not spans:
        return text
    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    pieces: list[str] = []
    position = 0
    for start, end in merged:
        pieces.append(text[position:start])
        position = max(position, end)
    pieces.append(text[position:])
    return "".join(pieces)
