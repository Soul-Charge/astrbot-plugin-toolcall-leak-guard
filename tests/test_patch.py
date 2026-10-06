"""main.py 的补丁自测：用真的 AstrBot LLMResponse，不联网、不碰生产数据。

用法：
    docker run --rm --entrypoint python -v <插件目录的父目录>:/work -w /work \
      soulter/astrbot:v4.28.1 astrbot_plugin_toolcall_leak_guard/tests/test_patch.py
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from astrbot.core.message.message_event_result import MessageChain  # noqa: E402
from astrbot.core.provider.entities import LLMResponse  # noqa: E402

from astrbot_plugin_toolcall_leak_guard import main as guard_main  # noqa: E402
from astrbot_plugin_toolcall_leak_guard.parser import DSML_MARKER as M  # noqa: E402


class FakeToolSet:
    def __init__(self, names):
        self._names = list(names)

    def names(self):
        return list(self._names)


def leak_text(name: str = "web_search_tavily", query: str = "fable") -> str:
    return (
        f"<{M} calls>\n"
        f'<{M} invoke name="{name}">\n'
        f'<{M} parameter name="query" string="true">{query}</{M} parameter>\n'
        f'<{M} parameter name="max_results" string="false">8</{M} parameter>\n'
        f"</{M} invoke>\n"
        f"</{M} calls>"
    )


def make_response(text: str) -> LLMResponse:
    resp = LLMResponse("assistant")
    resp.result_chain = MessageChain().message(text)
    resp.id = "test-response"
    return resp


def make_provider(response: LLMResponse):
    class FakeProvider:
        async def _parse_openai_completion(self, completion, tools=None):
            return response

    return FakeProvider


def run(response: LLMResponse, tools, **settings):
    guard = guard_main.GUARD
    saved = {
        key: getattr(guard, key)
        for key in (
            "enable",
            "convert",
            "allow_with_leftover_text",
            "strip_when_not_convertible",
            "dry_run",
            "log_raw",
        )
    }
    for key, value in settings.items():
        setattr(guard, key, value)
    try:
        cls = make_provider(response)
        assert guard_main.apply_parse_patch(cls) == "patched"
        provider = cls()
        return asyncio.run(provider._parse_openai_completion(None, tools))
    finally:
        for key, value in saved.items():
            setattr(guard, key, value)


def test_converts_and_clears_text() -> None:
    resp = run(make_response(leak_text()), FakeToolSet(["web_search_tavily"]))
    assert resp.tools_call_name == ["web_search_tavily"], resp.tools_call_name
    assert resp.tools_call_args == [{"query": "fable", "max_results": 8}], resp.tools_call_args
    assert len(resp.tools_call_ids) == 1 and resp.tools_call_ids[0]
    assert resp.role == "tool"
    assert resp.completion_text == "", repr(resp.completion_text)
    assert resp.result_chain is None
    assert resp.id == "test-response"


def test_two_calls_get_unique_ids() -> None:
    text = f"<{M} calls>\n" + (
        f'<{M} invoke name="web_search_tavily">\n'
        f'<{M} parameter name="query" string="true">a</{M} parameter>\n'
        f"</{M} invoke>\n"
        f'<{M} invoke name="web_search_tavily">\n'
        f'<{M} parameter name="query" string="true">b</{M} parameter>\n'
        f"</{M} invoke>\n"
    ) + f"</{M} calls>"
    resp = run(make_response(text), FakeToolSet(["web_search_tavily"]))
    assert len(resp.tools_call_ids) == 2
    assert resp.tools_call_ids[0] != resp.tools_call_ids[1]


def test_unknown_tool_only_strips() -> None:
    resp = run(make_response(leak_text()), FakeToolSet(["search_memes"]))
    assert resp.tools_call_name == []
    assert resp.completion_text == ""
    assert resp.result_chain is None


def test_no_tools_only_strips() -> None:
    resp = run(make_response(leak_text()), None)
    assert resp.tools_call_name == []
    assert resp.completion_text == ""


def test_keeps_prose_and_does_not_convert_by_default() -> None:
    text = "喵，窝去翻翻看~\n" + leak_text()
    resp = run(make_response(text), FakeToolSet(["web_search_tavily"]))
    assert resp.tools_call_name == [], "块外有正文时默认不还原"
    assert "喵，窝去翻翻看~" in resp.completion_text
    assert "parameter" not in resp.completion_text


def test_allow_with_leftover_converts_and_keeps_prose() -> None:
    text = "喵，窝去翻翻看~\n" + leak_text()
    resp = run(
        make_response(text),
        FakeToolSet(["web_search_tavily"]),
        allow_with_leftover_text=True,
    )
    assert resp.tools_call_name == ["web_search_tavily"]
    assert resp.completion_text == "喵，窝去翻翻看~"


def test_native_tool_call_is_untouched_but_text_stripped() -> None:
    resp = make_response(leak_text())
    resp.role = "tool"
    resp.tools_call_name = ["web_search_tavily"]
    resp.tools_call_args = [{"query": "native"}]
    resp.tools_call_ids = ["call_native_1"]
    out = run(resp, FakeToolSet(["web_search_tavily"]))
    assert out.tools_call_ids == ["call_native_1"]
    assert out.tools_call_args == [{"query": "native"}]
    assert out.completion_text == ""


def test_plain_text_untouched() -> None:
    resp = run(make_response("喵~ 今天也很乖"), FakeToolSet(["web_search_tavily"]))
    assert resp.completion_text == "喵~ 今天也很乖"
    assert resp.tools_call_name == []


def test_dry_run_does_not_change() -> None:
    resp = run(
        make_response(leak_text()),
        FakeToolSet(["web_search_tavily"]),
        dry_run=True,
    )
    assert resp.tools_call_name == []
    assert resp.completion_text.startswith("<" + M), "dry-run 不应改动响应"


def test_convert_disabled_only_strips() -> None:
    resp = run(
        make_response(leak_text()),
        FakeToolSet(["web_search_tavily"]),
        convert=False,
    )
    assert resp.tools_call_name == []
    assert resp.completion_text == ""


def test_enable_disabled_passes_through() -> None:
    text = leak_text()
    resp = run(make_response(text), FakeToolSet(["web_search_tavily"]), enable=False)
    assert resp.completion_text == text


def test_plain_components_are_preserved() -> None:
    import astrbot.core.message.components as Comp

    resp = make_response("前缀 " + leak_text())
    assert resp.result_chain is not None
    resp.result_chain.chain.append(Comp.Plain("尾巴"))
    out = run(resp, FakeToolSet(["web_search_tavily"]))
    assert "前缀" in out.completion_text
    assert "尾巴" in out.completion_text


def test_real_provider_class_can_be_patched() -> None:
    state = guard_main.apply_parse_patch(guard_main.ProviderOpenAIOfficial)
    assert state in ("patched", "already"), state
    target = getattr(
        guard_main.ProviderOpenAIOfficial,
        guard_main.TARGET_METHOD,
    )
    assert getattr(target, guard_main.PATCH_ATTR) == guard_main.PATCH_VERSION


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            import traceback

            print(f"FAIL {fn.__name__}: {exc!r}")
            traceback.print_exc()
        else:
            print(f"PASS {fn.__name__}")
    print(f"---- {len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
