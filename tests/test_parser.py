"""parser.py 的自测脚本（用容器里的 python 直接跑，不需要 pytest）。

用法：
    docker exec astrbot python /AstrBot/data/plugins/astrbot_plugin_toolcall_leak_guard/tests/test_parser.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parser import (  # noqa: E402
    DSML_MARKER as M,
    extract_leaked_tool_calls,
    looks_like_leak,
)


def calls_open() -> str:
    return f"<{M} calls>"


def calls_close() -> str:
    return f"</{M} calls>"


def inv_open(name: str) -> str:
    return f'<{M} invoke name="{name}">'


def inv_close() -> str:
    return f"</{M} invoke>"


def par_open(name: str, string_flag: str | None = "true") -> str:
    if string_flag is None:
        return f'<{M} parameter name="{name}">'
    return f'<{M} parameter name="{name}" string="{string_flag}">'


def par_close() -> str:
    return f"</{M} parameter>"


def test_search_memes_sample() -> None:
    text = (
        calls_open()
        + "\n"
        + inv_open("search_memes")
        + "\n"
        + par_open("query")
        + "被捏脸揉脸委屈炸毛撒娇"
        + par_close()
        + "\n"
        + inv_close()
        + "\n"
        + calls_close()
    )
    assert looks_like_leak(text)
    res = extract_leaked_tool_calls(text)
    assert res.found is True and res.unparsed is False, res
    assert res.leftover.strip() == "", repr(res.leftover)
    assert [c.name for c in res.calls] == ["search_memes"], res.calls
    assert res.calls[0].args == {"query": "被捏脸揉脸委屈炸毛撒娇"}, res.calls[0].args


def test_two_tavily_calls_and_int_arg() -> None:
    text = (
        calls_open()
        + "\n"
        + inv_open("web_search_tavily")
        + "\n"
        + par_open("query")
        + "Anthropic Claude Fable model release"
        + par_close()
        + "\n"
        + par_open("max_results", "false")
        + "8"
        + par_close()
        + "\n"
        + inv_close()
        + "\n"
        + inv_open("web_search_tavily")
        + "\n"
        + par_open("query")
        + 'Anthropic "Fable" 模型 发布'
        + par_close()
        + "\n"
        + par_open("max_results", "false")
        + "8"
        + par_close()
        + "\n"
        + inv_close()
        + "\n"
        + calls_close()
    )
    res = extract_leaked_tool_calls(text)
    assert [c.name for c in res.calls] == ["web_search_tavily", "web_search_tavily"]
    assert res.calls[0].args["max_results"] == 8
    assert isinstance(res.calls[0].args["max_results"], int)
    assert res.calls[1].args["query"] == 'Anthropic "Fable" 模型 发布'


def test_caption_sample_multiline_and_json_array() -> None:
    caption = "嘴上说着“对不起”，脸上却闭眼笑、抬手随意一挥。\n常用于被要求帮忙时。"
    text = (
        calls_open()
        + "\n"
        + inv_open("submit_meme_caption")
        + "\n"
        + par_open("caption")
        + caption
        + par_close()
        + "\n"
        + par_open("tags", "false")
        + '["对不起", "做不到", "摆烂"]'
        + par_close()
        + "\n"
        + par_open("visible_text")
        + "对不起 做不到"
        + par_close()
        + "\n"
        + inv_close()
        + "\n"
        + calls_close()
    )
    res = extract_leaked_tool_calls(text)
    assert len(res.calls) == 1, res.calls
    args = res.calls[0].args
    assert args["tags"] == ["对不起", "做不到", "摆烂"], args
    assert args["caption"] == caption, repr(args["caption"])
    assert args["visible_text"] == "对不起 做不到"


def test_plain_text_untouched() -> None:
    for text in (
        "",
        "喵~ 窝去翻翻看~",
        "我刚才想调用 invoke 那个工具，但是算了",
        "1 加 1 等于 2，没有别的",
    ):
        res = extract_leaked_tool_calls(text)
        assert res.found is False, (text, res)
        assert res.leftover == text
        assert res.calls == []


def test_prose_around_block_is_kept() -> None:
    text = (
        "喵，窝去翻翻看~\n"
        + calls_open()
        + "\n"
        + inv_open("web_search_tavily")
        + "\n"
        + par_open("query")
        + "test"
        + par_close()
        + "\n"
        + inv_close()
        + "\n"
        + calls_close()
        + "\n稍等喵~"
    )
    res = extract_leaked_tool_calls(text)
    assert res.found is True
    assert len(res.calls) == 1
    assert "喵，窝去翻翻看~" in res.leftover
    assert "稍等喵~" in res.leftover
    assert "parameter" not in res.leftover


def test_truncated_block() -> None:
    text = (
        "喵~\n"
        + calls_open()
        + "\n"
        + inv_open("web_search_tavily")
        + "\n"
        + par_open("query")
        + "被截断了"
    )
    res = extract_leaked_tool_calls(text)
    assert res.found is True
    assert res.unparsed is True
    assert res.leftover.strip() == "喵~", repr(res.leftover)


def test_xml_entity_and_no_marker_variant() -> None:
    text = (
        "<calls>\n"
        '<invoke name="echo">\n'
        '<parameter name="text" string="true">a &amp; b &lt;c&gt;</parameter>\n'
        "</invoke>\n"
        "</calls>"
    )
    res = extract_leaked_tool_calls(text)
    assert res.found is True, res
    assert res.calls[0].args["text"] == "a & b <c>", res.calls[0].args


def test_unclosed_invoke_loose_params() -> None:
    text = (
        calls_open()
        + "\n"
        + inv_open("web_search_tavily")
        + "\n"
        + par_open("query")
        + "abc"
        + par_close()
        + "\n"
    )
    res = extract_leaked_tool_calls(text)
    assert res.found is True
    assert res.unparsed is True or res.calls, res


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {exc!r}")
        else:
            print(f"PASS {fn.__name__}")
    print(f"---- {len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
