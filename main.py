"""astrbot_plugin_toolcall_leak_guard —— 把泄漏成文本的工具调用还原成原生调用。

问题
----
上游 chat endpoint（本机为 deepseek/deepseek-flash）偶尔不返回原生 tool_calls，
而是把工具调用写成一段文本塞进 message.content：标签由 calls / invoke /
parameter 三个词组成，每个标签前还带着模型自己的特殊标记（两个全角竖线 +
DSML + 两个全角竖线）。AstrBot 的 OpenAI 适配器只认原生 tool_calls，于是这段
标签文本被当成回复原文直接发进 QQ 群：工具没执行，人看到的是一堆代码。

方案
----
插件加载时给 ProviderOpenAIOfficial._parse_openai_completion 打一层后置处理：

1. 命中这类标签块 -> 先从要发出的正文里摘掉（绝不漏进群）。
2. 工具名在本次请求的工具集里、且块外没有别的正文 -> 还原成真正的 tool_calls
   （role / name / args / id 全部补齐），AstrBot 的 agent 循环会照常执行。
3. 解析失败、工具名对不上、或本次没有工具 -> 只摘除，并打 WARNING。

非流式（_query）和流式（_query_stream）最终都汇聚到这个函数，所以一处补丁
全覆盖；context.llm_generate(..., tools=...) 这类插件内部调用同样生效。

卸载插件即完全恢复原生行为（补丁只在插件加载时生效）。
"""

from __future__ import annotations

import uuid
from functools import wraps
from typing import Any

try:
    from .parser import extract_leaked_tool_calls, looks_like_leak
except ImportError:  # 非包方式加载（例如直接跑单测）
    from parser import (  # type: ignore[no-redef]
        extract_leaked_tool_calls,
        looks_like_leak,
    )

PLUGIN_NAME = "astrbot_plugin_toolcall_leak_guard"
DESCRIPTION = "模型把工具调用写成文本时，先摘掉不让它漏进群，再还原成原生工具调用继续执行"
PATCH_VERSION = "1.0.0"
PATCH_ATTR = "_toolcall_leak_guard_patch"
PATCH_ORIG_ATTR = "_toolcall_leak_guard_original"
TARGET_METHOD = "_parse_openai_completion"
LOG_TAG = "[toolcall_leak_guard]"

# 补丁目标（容器内一定存在；缺失时插件只打警告，不影响 AstrBot 运行）
try:
    from astrbot.api import logger
    from astrbot.api.star import Context, Star, register

    _ASTRBOT_AVAILABLE = True
except Exception:  # noqa: BLE001
    _ASTRBOT_AVAILABLE = False
    logger = None  # type: ignore[assignment]
    Context = object  # type: ignore[assignment,misc]
    Star = object  # type: ignore[assignment,misc]
    register = None  # type: ignore[assignment]

try:
    import astrbot.core.message.components as Comp
except Exception:  # noqa: BLE001
    Comp = None  # type: ignore[assignment]

try:
    from astrbot.core.provider.sources.openai_source import ProviderOpenAIOfficial
except Exception:  # noqa: BLE001
    ProviderOpenAIOfficial = None  # type: ignore[assignment]


def _log_info(message: str) -> None:
    if logger is not None:
        logger.info(message)


def _log_warning(message: str) -> None:
    if logger is not None:
        logger.warning(message)


def _log_error(message: str) -> None:
    if logger is not None:
        logger.error(message)


class _LeakGuard:
    """补丁管理 + 响应后处理。"""

    def __init__(self) -> None:
        self.enable = True
        self.convert = True
        self.allow_with_leftover_text = False
        self.strip_when_not_convertible = True
        self.dry_run = False
        self.log_raw = True
        self.stats = {"hits": 0, "converted": 0, "stripped": 0, "skipped": 0}

    # ------------------------------------------------------------------ 配置
    def refresh_config(self, config: Any) -> None:
        config = config if isinstance(config, dict) else {}

        def _flag(key: str, default: bool) -> bool:
            value = config.get(key, default)
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "on")
            return bool(value)

        self.enable = _flag("enable", True)
        self.convert = _flag("convert", True)
        self.allow_with_leftover_text = _flag("allow_with_leftover_text", False)
        self.strip_when_not_convertible = _flag("strip_when_not_convertible", True)
        self.dry_run = _flag("dry_run", False)
        self.log_raw = _flag("log_raw", True)
        _log_info(
            f"{LOG_TAG} 配置已加载: enable={self.enable} convert={self.convert} "
            f"allow_with_leftover_text={self.allow_with_leftover_text} "
            f"strip_when_not_convertible={self.strip_when_not_convertible} "
            f"dry_run={self.dry_run}"
        )

    # -------------------------------------------------------------- 后处理
    def postprocess(self, response: Any, tools: Any) -> Any:
        try:
            if not self.enable:
                return response
            text = getattr(response, "completion_text", None) or ""
            if not text or not looks_like_leak(text):
                return response

            parsed = extract_leaked_tool_calls(text)
            if not parsed.found:
                return response

            self.stats["hits"] += 1
            native_names = list(getattr(response, "tools_call_name", None) or [])
            known = self._tool_names(tools)
            valid = [call for call in parsed.calls if call.name in known] if known else []
            leftover = parsed.leftover.strip()

            convertible = (
                self.convert
                and not native_names
                and bool(known)
                and bool(valid)
                and len(valid) == len(parsed.calls)
                and (not leftover or self.allow_with_leftover_text)
            )

            if self.dry_run:
                self.stats["skipped"] += 1
                _log_info(
                    f"{LOG_TAG} DRY-RUN 命中文本形式的工具调用: "
                    f"parsed={[call.name for call in parsed.calls]} "
                    f"valid={[call.name for call in valid]} native={native_names} "
                    f"leftover={len(leftover)}"
                )
            elif convertible:
                self._apply_conversion(response, valid, leftover)
                self.stats["converted"] += 1
                _log_info(
                    f"{LOG_TAG} 已把文本形式的工具调用还原为原生调用: "
                    f"{[call.name for call in valid]}"
                )
            elif self.strip_when_not_convertible:
                self._set_text(response, leftover)
                self.stats["stripped"] += 1
                _log_warning(
                    f"{LOG_TAG} 已摘除泄漏的工具调用文本但未还原: "
                    f"parsed={[call.name for call in parsed.calls]} "
                    f"valid={[call.name for call in valid]} native={native_names} "
                    f"unparsed={parsed.unparsed} leftover={len(leftover)} "
                    f"known_tools={len(known)}"
                )
            else:
                self.stats["skipped"] += 1
                _log_warning(f"{LOG_TAG} 命中泄漏文本，但按配置未做任何处理")

            if self.log_raw and not self.dry_run:
                _log_info(f"{LOG_TAG} 原始文本(截断 800 字): {text[:800]!r}")
            return response
        except Exception as exc:  # noqa: BLE001 - 绝不让插件本身弄挂请求
            _log_error(f"{LOG_TAG} 处理响应时出错，已放行原始响应: {exc}")
            return response

    @staticmethod
    def _tool_names(tools: Any) -> set[str]:
        if tools is None:
            return set()
        try:
            names = tools.names()
        except Exception:  # noqa: BLE001
            return set()
        return {str(name) for name in (names or []) if name}

    @staticmethod
    def _apply_conversion(response: Any, calls: list[Any], leftover: str) -> None:
        response.role = "tool"
        response.tools_call_name = [call.name for call in calls]
        response.tools_call_args = [call.args for call in calls]
        response.tools_call_ids = [
            f"call_leak_{uuid.uuid4().hex[:20]}" for _ in calls
        ]
        response.tools_call_extra_content = {}
        _LeakGuard._set_text(response, leftover)

    @staticmethod
    def _set_text(response: Any, text: str) -> None:
        """替换响应的正文，同时不破坏正文之外的消息组件。"""
        chain = getattr(response, "result_chain", None)
        if chain is None:
            try:
                response._completion_text = text  # noqa: SLF001
            except Exception:  # noqa: BLE001
                response.completion_text = text
            return

        keep = []
        if Comp is not None:
            keep = [
                comp
                for comp in list(getattr(chain, "chain", None) or [])
                if not isinstance(comp, Comp.Plain)
            ]
        if text and Comp is not None:
            keep.insert(0, Comp.Plain(text))

        if keep:
            try:
                response.result_chain = chain.derive(keep)
                return
            except Exception:  # noqa: BLE001
                pass
        response.result_chain = None
        try:
            response._completion_text = text  # noqa: SLF001
        except Exception:  # noqa: BLE001
            response.completion_text = text


GUARD = _LeakGuard()


def apply_parse_patch(provider_cls: Any) -> str:
    """给 provider 类的解析方法打后置处理补丁。

    Returns:
        'patched' | 'already' | 'missing'
    """
    if provider_cls is None:
        return "missing"
    target = getattr(provider_cls, TARGET_METHOD, None)
    if target is None:
        return "missing"
    if getattr(target, PATCH_ATTR, None) == PATCH_VERSION:
        return "already"

    original = getattr(target, PATCH_ORIG_ATTR, None) or target

    @wraps(original)
    async def _parse_openai_completion(self, completion, tools=None):
        response = await original(self, completion, tools)
        return GUARD.postprocess(response, tools)

    setattr(_parse_openai_completion, PATCH_ATTR, PATCH_VERSION)
    setattr(_parse_openai_completion, PATCH_ORIG_ATTR, original)
    setattr(provider_cls, TARGET_METHOD, _parse_openai_completion)
    return "patched"


if not _ASTRBOT_AVAILABLE:  # pragma: no cover - 仅用于脱离 AstrBot 的静态检查
    __all__ = ["GUARD", "_LeakGuard", "apply_parse_patch"]
else:

    @register(PLUGIN_NAME, "NaE", DESCRIPTION, PATCH_VERSION)
    class ToolcallLeakGuardPlugin(Star):
        def __init__(self, context: Context, config: dict):
            super().__init__(context)
            GUARD.refresh_config(config)
            self._apply_patches(context)

        def _apply_patches(self, context: Context) -> None:
            results: dict[str, str] = {}
            results["ProviderOpenAIOfficial"] = apply_parse_patch(
                ProviderOpenAIOfficial
            )

            providers = []
            try:
                providers = list(context.get_all_providers() or [])
            except Exception as exc:  # noqa: BLE001
                _log_warning(f"{LOG_TAG} 获取提供商列表失败: {exc}")

            for provider in providers:
                try:
                    cls = type(provider)
                    key = f"{cls.__module__}.{cls.__qualname__}"
                    if key not in results:
                        results[key] = apply_parse_patch(cls)
                except Exception:  # noqa: BLE001
                    continue

            _log_info(f"{LOG_TAG} 补丁状态: {results}")
            if results.get("ProviderOpenAIOfficial") == "missing":
                _log_warning(
                    f"{LOG_TAG} 没找到 {TARGET_METHOD}，补丁未生效"
                    "（AstrBot 可能升级改动了该函数，需要复核）"
                )

    __all__ = [
        "GUARD",
        "ToolcallLeakGuardPlugin",
        "_LeakGuard",
        "apply_parse_patch",
    ]
