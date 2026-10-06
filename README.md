# astrbot_plugin_toolcall_leak_guard

把「模型写成文本的工具调用」摘掉并还原成原生工具调用。

## 现象

上游 chat endpoint（本机是 deepseek/deepseek-flash）偶尔不返回原生 tool_calls，
而是把调用写成一段文本塞进 message.content。日志里的原始样子：

    completion: ChatCompletion(... finish_reason='stop' ...,
      message=ChatCompletionMessage(
        content='...calls 标签...invoke name="web_search_tavily"...parameter...',
        tool_calls=None))

AstrBot 的 OpenAI 适配器只解析原生 tool_calls，于是这段标签文本被当成回复原文
直接发进群，工具也没有执行。2026-09-16 01:37 / 18:37 / 19:21 三次都是这个原因。

## 方案

插件加载时给 ProviderOpenAIOfficial._parse_openai_completion 打一层后置处理：

1. 命中这类标签块 -> 先从要发出的正文里摘掉（绝不漏进群）。
2. 工具名在本次请求的工具集里、且块外没有别的正文 -> 还原成真正的 tool_calls
   （role / name / args / id 全部补齐），AstrBot 的 agent 循环照常执行。
3. 解析失败、工具名对不上、本次没有工具 -> 只摘除，并打 WARNING 日志。

非流式（_query）和流式（_query_stream）最终都汇聚到这个函数，一处补丁全覆盖；
context.llm_generate(..., tools=...) 这类插件内部调用同样生效。

## 配置（WebUI -> 插件管理 -> 本插件）

- enable：总开关。关掉即完全恢复 AstrBot 原生行为。
- convert：命中时是否真的还原并执行。关掉就只摘文本、不执行。
- allow_with_leftover_text：默认 false，只处理「整条回复就是调用块」的情况。
  放宽后块外的正文会被当作「噪声」一起交给工具结果流程，可能吞掉正常解释。
- strip_when_not_convertible：默认 true。工具名对不上/没工具/解析失败时也摘掉。
- dry_run：演习模式，只记日志不改响应。上线初期想只看命中率时打开。
- log_raw：命中时把原始文本截断打到日志（INFO）。

## 验证

1. 插件加载后日志里应该有：

       [toolcall_leak_guard] 配置已加载: ...
       [toolcall_leak_guard] 补丁状态: {'ProviderOpenAIOfficial': 'patched', ...}

   如果看到 missing，说明 AstrBot 升级改动了目标函数名，补丁没生效，需要复核。

2. 之后每次命中会有：

       [toolcall_leak_guard] 已把文本形式的工具调用还原为原生调用: ['web_search_tavily']

   或者

       [toolcall_leak_guard] 已摘除泄漏的工具调用文本但未还原: ...

   同时群里不再出现带尖括号的那段文本。

## 单元测试

    docker run --rm --entrypoint python -v <插件目录的父目录>:/work -w /work \
      soulter/astrbot:v4.28.1 astrbot_plugin_toolcall_leak_guard/tests/test_parser.py

测试只覆盖纯文本解析（parser.py），不碰 AstrBot、不碰生产数据。

## 回滚

- WebUI 里禁用/卸载插件即可，补丁只在插件加载时生效。
- 或把插件目录移走（按项目规矩先移到 F:\AiHarness\Trash\，不要直接删）。
- 补丁带幂等标记，重复加载不会叠加多层包装。

## 注意

- 流式（streaming_response=true）时，标签文本可能已经在增量阶段发给平台了，
  本补丁只能保证最终响应文本干净。本机当前是流式关闭，无影响。
- 这是治标：根因是上游 endpoint 间歇性不返回原生 tool_calls。想彻底解决要换
  一个能稳定返回原生工具调用的通道，或者推动上游修。
