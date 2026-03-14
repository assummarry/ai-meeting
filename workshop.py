"""
多智能体白皮书工坊 (Multi-Agent Whitepaper Workshop) — v4
=====================================================================
技术栈: Python 3.10+, LangGraph >=0.2.0, Streamlit, OpenAI SDK (兼容 DeepSeek/OpenAI)

四车间流水线:
  1. 架构师内阁 — 主持人动态点名(图驱动单步) + 三位架构师脑暴/修复
  2. 文档压制   — 将讨论凝炼为结构化白皮书 JSON → Markdown
  3. 红蓝对抗   — 挑刺师1(致命 Fail-Fast) + 挑刺师2(次要)
  4. 全局仲裁   — 退火策略决定 "打回重做" 或 "汇报老板"

v2/v3 修复（全部保留）：
  ✅ [fix1] 双 Buffer 架构      : round_messages + display_messages
  ✅ [fix2][fix6] Token 软截断  : CJK×1.0 + ASCII×0.3 精化估算
  ✅ [fix3] 真物理隔断          : smart_add + 哨兵值清空
  ✅ [fix4] 模型兼容性三层降级  : response_format → Prompt → 正则
  ✅ [fix5] 并发状态穿透隔离    : JSON 缓存移入 session_state
  ✅ [fix7] 死循环熔断          : cabinet_call_count 超限强制 consensus

v4 新增：
  ✅ [feat1] 流式输出           : 架构师发言改为 stream=True 自由文本，
           实时打字机效果；主持人路由决策仍保持 JSON（稳定路由不受影响）
  ✅ [feat2] 多 Session 管理   : 新建/切换/删除话题，Markdown 导出，
           自动标题生成，页面刷新后历史不丢失
  ✅ [feat3] 摘要长程记忆       : workshop_2 清空 round_messages 前先
           LLM 压缩摘要存入 summary_context，下一轮架构师 prompt 注入，
           解决"修复 A 忘记 B"的跨轮遗忘
  ✅ [feat4] 输出长度控制       : 侧边栏简洁/标准/详细，注入架构师
           system prompt，节省 Token 和等待时间

运行方式:
  pip install langgraph streamlit openai
  streamlit run workshop_v4.py
"""

from __future__ import annotations

import json
import operator
import re
import textwrap
import time
from typing import Annotated, List, TypedDict

import streamlit as st
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from openai import OpenAI


# ═══════════════════════════════════════════════════════════════════════════════
#  [fix3] 自定义 Reducer: smart_add
#
#  operator.add 只能追加，无法清空列表。
#  smart_add 在保持追加语义的同时，支持通过哨兵值 {"__clear__": True}
#  触发清空，实现真正的物理隔断。
#
#  节点返回 [{"__clear__": True}]  →  状态清空为 []
#  节点返回普通消息列表            →  追加（等同 operator.add）
# ═══════════════════════════════════════════════════════════════════════════════

def smart_add(left: List[dict], right: List[dict]) -> List[dict]:
    if any(m.get("__clear__") for m in right):
        # 过滤哨兵，保留同批次其他消息（通常为空）
        remainder = [m for m in right if not m.get("__clear__")]
        return remainder
    return left + right


# ═══════════════════════════════════════════════════════════════════════════════
#  [fix1] 双 Buffer 状态设计
#
#  ┌─────────────────┬─────────────────────────────────────────────────────┐
#  │ round_messages  │ 当前脑暴轮消息，打回重做时由 smart_add 清空         │
#  │                 │ LLM Prompt 组装只从此字段取上下文                   │
#  │ display_messages│ 仅追加，供 Streamlit UI 渲染，永不删除              │
#  └─────────────────┴─────────────────────────────────────────────────────┘
# ═══════════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    round_messages:   Annotated[List[dict], smart_add]        # 当前轮，支持清空
    display_messages: Annotated[List[dict], operator.add]     # UI 专用，只追加
    whitepaper:        str
    feedback_fatal:    str
    feedback_minor:    str
    loop_count:        int
    consensus_reached: bool
    final_decision:    str
    user_prompt:       str
    arbitration_reason: str
    cabinet_call_count: int   # [fix7] 单轮脑暴内主持人 call 次数，超限强制 consensus
    summary_context:   str    # [feat3] 跨轮滚动摘要，防止遗忘


# ═══════════════════════════════════════════════════════════════════════════════
#  [fix2][fix6] 精化 Token 估算的软截断工具
#
#  旧版 len(text)//2 对纯英文/代码段高估近一倍，导致上下文过早截断。
#  改进：中英文分别计算，更准确地逼近实际 Token 消耗：
#    - CJK 汉字：约 1.0 token/字符
#    - ASCII 字母/数字/符号：约 0.3 token/字符（4 字符≈1 token）
#  无需引入 tiktoken 等额外依赖。
# ═══════════════════════════════════════════════════════════════════════════════

_RE_CJK = re.compile(r'[一-鿿㐀-䶿豈-﫿]')

def _estimate_tokens(text: str) -> int:
    """混合中英文 Token 估算：CJK×1.0 + ASCII×0.3，向上取整。"""
    if not text:
        return 1
    cjk_count   = len(_RE_CJK.findall(text))
    other_count = len(text) - cjk_count
    return max(1, int(cjk_count * 1.0 + other_count * 0.3))


def trim_by_token_budget(messages: List[dict], token_budget: int = 3000) -> List[dict]:
    """从最新消息向前，贪心保留不超过预算的消息，返回时序保持旧→新。"""
    selected: List[dict] = []
    used = 0
    for msg in reversed(messages):
        cost = _estimate_tokens(msg.get("content", ""))
        if used + cost > token_budget:
            break
        selected.append(msg)
        used += cost
    return list(reversed(selected))


def fmt_msgs(messages: List[dict], token_budget: int = 3000) -> str:
    """截断后格式化为可嵌入 Prompt 的字符串。"""
    trimmed = trim_by_token_budget(messages, token_budget)
    return "\n".join(
        f"[{m.get('role', '?')}] {m.get('content', '')}" for m in trimmed
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  角色 System Prompts（模块级常量）
# ═══════════════════════════════════════════════════════════════════════════════

SYS_MODERATOR = textwrap.dedent("""\
    你是架构师内阁的最高决策主持人。
    你的职责是根据当前讨论进展，动态决定下一步行动。
    只输出 JSON，格式：
    {"action": "call"|"consensus", "target": "架构师A"|"架构师B"|"架构师C"|null, "reason": "简要理由"}
    - action=call 表示点名某位架构师发言（每轮最多点名 3 次后必须 consensus）。
    - action=consensus 表示讨论已充分，可以进入文档压制。
""")

SYS_ARCH_A = textwrap.dedent("""\
    你是主业务架构师（架构师A），专注核心业务逻辑与领域建模。
    请根据当前讨论上下文给出你的专业意见。
    直接输出意见文本，不要 JSON 格式，不要代码块包裹。
""")

SYS_ARCH_B = textwrap.dedent("""\
    你是数据/后端架构师（架构师B），专注数据库设计、API 接口、性能优化。
    请根据当前讨论上下文给出你的专业意见。
    直接输出意见文本，不要 JSON 格式，不要代码块包裹。
""")

SYS_ARCH_C = textwrap.dedent("""\
    你是前端/交互架构师（架构师C），专注用户体验、界面流程与前端架构。
    请根据当前讨论上下文给出你的专业意见。
    直接输出意见文本，不要 JSON 格式，不要代码块包裹。
""")

ARCH_PROMPTS = {"架构师A": SYS_ARCH_A, "架构师B": SYS_ARCH_B, "架构师C": SYS_ARCH_C}

SYS_DOC_COMPRESS = textwrap.dedent("""\
    你是高级技术文档压制专家。
    将架构师们的讨论精华提炼为结构化技术白皮书。
    严格输出 JSON，格式：
    {
        "core_flow": "核心业务流程描述",
        "data_structure": "数据结构与模型设计",
        "api_definition": "API 接口定义",
        "rejected_ideas": "被否决方案及原因"
    }
""")

SYS_RED_FATAL = textwrap.dedent("""\
    你是红队挑刺师1号（致命缺陷检测）。
    审查白皮书，找出致命级别问题（安全漏洞、逻辑矛盾、数据丢失风险、不可行方案等）。
    输出 JSON：{"fatal_issues": "致命问题描述，没有则为空字符串 ''"}
""")

SYS_BLUE_MINOR = textwrap.dedent("""\
    你是蓝队挑刺师2号（次要瑕疵检测）。
    审查白皮书，找出次要问题（命名不一致、文档遗漏、可优化点等）。
    输出 JSON：{"minor_issues": "次要问题描述，没有则为空字符串 ''"}
""")

SYS_ARBITRATOR = textwrap.dedent("""\
    你是全局仲裁官，掌握最终决策权。
    根据红蓝对抗反馈和当前迭代轮次，决定白皮书是否可以交付。
    输出 JSON：{"decision": "打回重做"|"汇报老板", "reason": "决策理由"}

    退火规则（必须严格遵守）：
    - 第 1-3 轮：任何瑕疵（致命或次要）都应打回重做
    - 第 4-6 轮：仅致命或严重瑕疵才打回，次要问题可放行
    - 第 7 轮及以上：除非存在致命缺陷，否则强制通过（汇报老板）
""")

# ─── [feat3] 摘要压缩 Prompt ──────────────────────────────────────────────────
SYS_SUMMARIZER = textwrap.dedent("""\
    你是高效的会议纪要员。
    请将以下架构讨论高度浓缩，提取：
    1. 用户的核心需求要点
    2. 已确认的架构决策
    3. 被明确否决的方案
    输出纯文本，控制在 300 字以内，作为后续讨论的背景记忆。
""")

# ─── [feat4] 输出长度指令 ─────────────────────────────────────────────────────
DETAIL_INSTRUCTIONS: dict[str, str] = {
    "简洁": "\n\n【输出长度要求】请控制在 150 字以内，只给出核心结论，省略推导过程。",
    "标准": "",   # 不注入，模型自主判断
    "详细": "\n\n【输出长度要求】请充分展开，覆盖所有相关细节、边界情况和设计权衡，不限字数。",
}


# ═══════════════════════════════════════════════════════════════════════════════
#  [fix4] 兼容性 LLM 调用：自动探测 → 三层降级
#
#  Layer 1: response_format=json_object  (OpenAI / DeepSeek 原生)
#  Layer 2: 遇 400/422 → 去掉参数，纯 Prompt 引导再调用
#  Layer 3: JSON 解析失败 → 正则提取 {...} 块兜底
#
#  首次 Layer 1 失败后，将模型写入 _JSON_MODE_UNSUPPORTED 缓存，
#  后续调用直接从 Layer 2 开始，避免每次多一次无效请求。
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_json_from_text(text: str) -> dict | None:
    """从非标准文本中提取第一个 JSON 对象，处理 markdown 代码块等包裹。"""
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        try:
            return json.loads(brace.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _get_json_mode_cache() -> set:
    """[fix5] 从 st.session_state 取 per-session JSON 模式不支持缓存。
    模块全局 set 在多用户并发时所有 session 共享，会导致用户A的降级
    错误地影响用户B。改用 session_state 实现会话隔离。
    """
    if "json_mode_unsupported" not in st.session_state:
        st.session_state.json_mode_unsupported = set()
    return st.session_state.json_mode_unsupported


def call_llm_json(
    client: OpenAI,
    model: str,
    system: str,
    user_content: str,
    temperature: float = 0.7,
) -> dict:
    """调用大模型并强制返回解析后的 dict，具备三层降级容错。"""
    if "JSON" not in system.upper():
        system += "\n请以 JSON 格式输出，不要输出任何其他内容。"

    messages_payload = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
    raw = ""

    try:
        # ── Layer 1: 结构化 JSON 模式 ──────────────────────────────────────
        _json_cache = _get_json_mode_cache()
        if model not in _json_cache:
            try:
                resp = client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                    messages=messages_payload,
                )
                raw = resp.choices[0].message.content.strip()
                return json.loads(raw)
            except Exception as e:
                err = str(e).lower()
                if any(k in err for k in ["400", "422", "response_format", "unsupported", "not support"]):
                    _json_cache.add(model)
                    # 降级到 Layer 2，不 re-raise
                else:
                    raise  # 网络/鉴权等真实错误，向上抛出

        # ── Layer 2: 纯 Prompt 引导，无 response_format ────────────────────
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=messages_payload,
        )
        raw = resp.choices[0].message.content.strip()

        # ── Layer 3: 解析 / 正则提取兜底 ──────────────────────────────────
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            extracted = _extract_json_from_text(raw)
            if extracted is not None:
                return extracted
            st.warning(f"⚠️ 无法从模型响应提取 JSON，已降级。原文片段: {raw[:200]}")
            return {"error": "无法解析 JSON", "raw": raw}

    except Exception as e:
        st.error(f"LLM 调用失败: {e}")
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
#  [feat1] 流式调用（架构师自由文本发言专用）
#
#  架构师的意见是自由文本，无需 JSON 解析，可以完全流式输出。
#  主持人的路由决策（call/consensus）仍走 call_llm_json，JSON 稳定不受影响。
#
#  参数 placeholder: 由调用方传入的 st.empty() 实例，控制渲染位置。
#  返回完整文本字符串，供写入 AgentState。
# ═══════════════════════════════════════════════════════════════════════════════

def call_llm_stream(
    client: OpenAI,
    model: str,
    system: str,
    user_content: str,
    placeholder,
    temperature: float = 0.7,
) -> str:
    """流式调用，实时更新 placeholder（打字机效果），返回完整文本。"""
    payload = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_content},
    ]
    full_text = ""
    try:
        stream = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=payload,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                full_text += delta
                placeholder.markdown(full_text + "▌")   # 打字机光标
        placeholder.markdown(full_text)                  # 最终渲染，去掉光标
    except Exception as e:
        full_text = f"⚠️ 流式调用失败: {e}"
        placeholder.markdown(full_text)
    return full_text


# ═══════════════════════════════════════════════════════════════════════════════
#  [feat3] 摘要压缩（非流式，后台静默，失败时静默返回空字符串）
# ═══════════════════════════════════════════════════════════════════════════════

def call_llm_summary(client: OpenAI, model: str, text: str) -> str:
    """将本轮讨论压缩为摘要，用于跨轮长程记忆。"""
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0.3,
            messages=[
                {"role": "system", "content": SYS_SUMMARIZER},
                {"role": "user",   "content": text},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
#  节点工厂（闭包注入 client & model）
# ═══════════════════════════════════════════════════════════════════════════════

def make_nodes(client: OpenAI, model: str, detail_instruction: str = "") -> dict:

    # ── 车间 1: 架构师内阁 ─────────────────────────────────────────────────────
    # [fix7] 单轮最多 call CABINET_CALL_LIMIT 次，超限强制 consensus，防止死循环
    CABINET_CALL_LIMIT = 5

    # ── [feat3] 摘要注入辅助 ──────────────────────────────────────────────────
    def _inject_summary(base_prompt: str, summary: str) -> str:
        """将跨轮摘要附加到 system prompt，有摘要才注入。"""
        if not summary:
            return base_prompt
        return (
            base_prompt
            + f"\n\n【跨轮历史摘要（请在此基础上延续，避免重蹈已否决方案）】\n{summary}"
        )

    def workshop_1_architect_cabinet(state: AgentState) -> dict:
        loop_count  = state.get("loop_count", 0)
        user_prompt = state.get("user_prompt", "")
        whitepaper  = state.get("whitepaper", "")
        fatal       = state.get("feedback_fatal", "")
        minor       = state.get("feedback_minor", "")
        call_count  = state.get("cabinet_call_count", 0)  # [fix7] 当前轮已 call 次数
        summary     = state.get("summary_context", "")    # [feat3] 跨轮摘要

        # [fix7] 强制熔断：超过上限直接 consensus，不再询问主持人
        if call_count >= CABINET_CALL_LIMIT:
            msg = {
                "role": "assistant",
                "content": (
                    f"🛑 **【系统熔断】** 本轮脑暴已达 {CABINET_CALL_LIMIT} 次点名上限，"
                    "强制进入文档压制，防止无限循环。"
                ),
            }
            return {
                "round_messages":    [msg],
                "display_messages":  [msg],
                "consensus_reached": True,
                "cabinet_call_count": 0,   # 重置，为下一轮准备
            }

        # [fix2][fix6] Token 预算截断取本轮上下文
        recent_ctx = fmt_msgs(state.get("round_messages", []), token_budget=2500)

        # [feat3] 主持人 prompt 注入摘要
        moderator_sys = _inject_summary(SYS_MODERATOR, summary)

        if loop_count == 0:
            moderator_input = (
                f"【脑暴模式】用户原始需求：\n{user_prompt}\n\n"
                f"本轮历史对话（已进行 {call_count} 次发言）：\n{recent_ctx}\n\n"
                f"剩余可点名次数：{CABINET_CALL_LIMIT - call_count} 次。"
                "请决定点名某位架构师发言，或判断讨论已充分可达成共识（consensus）。"
            )
        else:
            moderator_input = (
                f"【修复模式 · 第 {loop_count} 轮返工】\n"
                f"用户需求：{user_prompt}\n"
                f"致命问题：{fatal or '无'}\n次要问题：{minor or '无'}\n"
                f"白皮书摘要：\n{whitepaper[:600]}\n\n"
                f"本轮历史对话（已进行 {call_count} 次发言）：\n{recent_ctx}\n\n"
                f"剩余可点名次数：{CABINET_CALL_LIMIT - call_count} 次。"
                "请点名架构师局部修复，或判断已修复完毕达成共识（consensus）。"
            )

        mod_result = call_llm_json(client, model, moderator_sys, moderator_input)

        if "error" in mod_result:
            msg = {"role": "assistant", "content": "⚠️ **【主持人】** 调用异常，强制进入文档压制。"}
            return {
                "round_messages":    [msg],
                "display_messages":  [msg],
                "consensus_reached": True,
                "cabinet_call_count": 0,
            }

        action = mod_result.get("action", "consensus")
        target = mod_result.get("target")
        reason = mod_result.get("reason", "")

        if action == "consensus":
            msg = {"role": "assistant", "content": f"🎙️ **【主持人】** {reason}（达成共识，进入文档压制）"}
            return {
                "round_messages":    [msg],
                "display_messages":  [msg],
                "consensus_reached": True,
                "cabinet_call_count": 0,   # 重置
            }

        # ── [feat1] 架构师发言：流式输出 ──────────────────────────────────────
        # 架构师改为自由文本输出（已去掉 JSON 要求），可以完全流式渲染。
        # 先在 UI 打字机展示，完成后把完整文本写入 AgentState。
        arch_base_sys = ARCH_PROMPTS.get(target, SYS_ARCH_A)
        # [feat3][feat4] 注入摘要 + 长度控制
        arch_sys = _inject_summary(arch_base_sys, summary) + detail_instruction

        if loop_count == 0:
            arch_input = (
                f"用户需求：{user_prompt}\n\n"
                f"本轮历史对话：\n{recent_ctx}\n\n"
                "请基于上述背景提出你的架构设计意见。"
            )
        else:
            arch_input = (
                f"用户需求：{user_prompt}\n"
                f"【警告】上一版被打回，请仅针对以下痛点局部修复：\n"
                f"致命反馈：{fatal or '无'}\n次要反馈：{minor or '无'}\n"
                f"现有白皮书：\n{whitepaper}\n\n"
                f"本轮历史对话：\n{recent_ctx}"
            )

        host_msg = {"role": "assistant", "content": f"🎙️ **【主持人】** 呼叫 {target}。理由：{reason}"}

        # [feat1] 在 Streamlit 主线程中流式渲染架构师发言
        # LangGraph graph.stream() 在主线程调用节点，st.* 调用安全
        with st.chat_message("assistant"):
            st.markdown(host_msg["content"])
            st.markdown(f"🏗️ **【{target}】** 正在发言...")
            arch_placeholder = st.empty()

        opinion = call_llm_stream(
            client, model, arch_sys, arch_input, arch_placeholder
        )

        arch_msg = {"role": "assistant", "content": f"🏗️ **【{target}】** {opinion}"}
        # [feat1] __streamed__=True 标记：告知 UI 层此消息已流式直接渲染，跳过重复渲染
        arch_msg_display = {**arch_msg, "__streamed__": True}

        return {
            "round_messages":    [host_msg, arch_msg],
            "display_messages":  [host_msg, arch_msg_display],
            "consensus_reached": False,
            "cabinet_call_count": call_count + 1,        # [fix7] 递增，趋近熔断上限
        }

    # ── 车间 2: 文档压制 + [feat3] 摘要压缩 ───────────────────────────────────
    def workshop_2_doc_compression(state: AgentState) -> dict:
        user_prompt  = state.get("user_prompt", "")
        round_msgs   = state.get("round_messages", [])
        prev_summary = state.get("summary_context", "")

        # [feat3] 在清空 round_messages 之前，先压缩本轮讨论为摘要
        # 追加式：prev_summary + 本轮新摘要，实现滚动记忆
        new_summary = ""
        if round_msgs:
            text_to_compress = "\n".join(
                f"[{m.get('role','?')}] {m.get('content','')}" for m in round_msgs
            )
            new_summary = call_llm_summary(client, model, text_to_compress)

        if new_summary:
            summary_context = (
                f"{prev_summary}\n\n---（新一轮）---\n\n{new_summary}"
                if prev_summary else new_summary
            )
        else:
            summary_context = prev_summary

        # [fix2] 文档压制允许更大 Token 预算，尽量保留完整讨论
        discussion = fmt_msgs(round_msgs, token_budget=4000)

        user_input = (
            f"用户原始需求：\n{user_prompt}\n\n"
            f"架构师讨论记录：\n{discussion}\n\n"
            "请提炼为结构化技术白皮书。"
        )
        result = call_llm_json(client, model, SYS_DOC_COMPRESS, user_input, temperature=0.3)

        wp_md = textwrap.dedent(f"""\
        # 技术白皮书

        ## 核心业务流程
        {result.get("core_flow", "（未生成）")}

        ## 数据结构与模型
        {result.get("data_structure", "（未生成）")}

        ## API 接口定义
        {result.get("api_definition", "（未生成）")}

        ## 被否决方案
        {result.get("rejected_ideas", "（无）")}
        """)

        notice_msg = {"role": "assistant", "content": "📄 **白皮书已生成/更新**（详见下方展开区域）"}

        return {
            # [fix3] 真物理隔断：哨兵触发 smart_add 清空 round_messages
            "round_messages":   [{"__clear__": True}],
            "display_messages": [notice_msg],
            "whitepaper":       wp_md,
            "summary_context":  summary_context,   # [feat3] 更新跨轮摘要
        }

    # ── 车间 3: 红蓝对抗 ───────────────────────────────────────────────────────
    def workshop_3_red_blue_test(state: AgentState) -> dict:
        whitepaper = state.get("whitepaper", "")

        red_result = call_llm_json(
            client, model, SYS_RED_FATAL,
            f"请审查以下白皮书的致命缺陷：\n\n{whitepaper}",
            temperature=0.2,
        )
        fatal = red_result.get("fatal_issues", "").strip()

        if fatal:
            msg = {"role": "assistant", "content": f"🔴 **挑刺师1（致命）**：{fatal}\n⚡ Fail-Fast，跳过次要检测。"}
            return {"feedback_fatal": fatal, "feedback_minor": "", "display_messages": [msg]}

        blue_result = call_llm_json(
            client, model, SYS_BLUE_MINOR,
            f"请审查以下白皮书的次要问题：\n\n{whitepaper}",
            temperature=0.2,
        )
        minor = blue_result.get("minor_issues", "").strip()

        msgs = [
            {"role": "assistant", "content": "🔴 **挑刺师1（致命）**：未发现致命缺陷 ✅"},
            {"role": "assistant", "content": f"🔵 **挑刺师2（次要）**：{minor if minor else '未发现次要问题 ✅'}"},
        ]
        return {"feedback_fatal": "", "feedback_minor": minor, "display_messages": msgs}

    # ── 车间 4: 全局仲裁与退火 ─────────────────────────────────────────────────
    def workshop_4_global_arbitration(state: AgentState) -> dict:
        loop  = state.get("loop_count", 0)
        fatal = state.get("feedback_fatal", "")
        minor = state.get("feedback_minor", "")

        result = call_llm_json(
            client, model, SYS_ARBITRATOR,
            (
                f"当前迭代轮次：第 {loop + 1} 轮\n"
                f"致命问题反馈：{fatal or '无'}\n"
                f"次要问题反馈：{minor or '无'}\n\n"
                "请根据退火规则做出决策。"
            ),
            temperature=0.1,
        )

        decision = result.get("decision", "打回重做")
        reason   = result.get("reason", "未提供理由")

        # 系统强制兜底，防止 LLM 误判
        if loop + 1 >= 7 and not fatal:
            decision = "汇报老板"
            reason += f" [系统强制：已达第 {loop + 1} 轮且无致命缺陷，强制收敛]"
        elif loop + 1 <= 3 and (fatal or minor):
            decision = "打回重做"
            reason += " [系统强制：早期严苛期，存在任何瑕疵强制打回]"

        msg = {
            "role": "assistant",
            "content": (
                f"⚖️ **全局仲裁（第 {loop + 1} 轮）**：**{decision}**\n"
                f"理由：{reason}"
            ),
        }
        return {
            "final_decision":    decision,
            "loop_count":        loop + 1,
            "arbitration_reason": reason,
            "display_messages":  [msg],
        }

    # ── 重置节点 ───────────────────────────────────────────────────────────────
    def reset_feedback_node(state: AgentState) -> dict:
        return {
            "feedback_fatal":    "",
            "feedback_minor":    "",
            "consensus_reached": False,
            "cabinet_call_count": 0,           # [fix7] 打回后重置点名计数
            "round_messages":    [{"__clear__": True}],  # [fix3] 清空本轮消息
        }

    return {
        "workshop_1":    workshop_1_architect_cabinet,
        "workshop_2":    workshop_2_doc_compression,
        "workshop_3":    workshop_3_red_blue_test,
        "workshop_4":    workshop_4_global_arbitration,
        "reset_feedback": reset_feedback_node,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  路由函数
# ═══════════════════════════════════════════════════════════════════════════════

def route_after_cabinet(state: AgentState) -> str:
    return "workshop_2" if state.get("consensus_reached") else "workshop_1"


def route_after_arbitration(state: AgentState) -> str:
    return END if state.get("final_decision") == "汇报老板" else "reset_feedback"


# ═══════════════════════════════════════════════════════════════════════════════
#  构建 LangGraph StateGraph
# ═══════════════════════════════════════════════════════════════════════════════

def build_graph(client: OpenAI, model: str, detail_instruction: str = "") -> CompiledStateGraph:
    nodes = make_nodes(client, model, detail_instruction)
    builder = StateGraph(AgentState)

    for name, func in nodes.items():
        builder.add_node(name, func)

    builder.add_edge(START, "workshop_1")
    builder.add_conditional_edges(
        "workshop_1",
        route_after_cabinet,
        {"workshop_1": "workshop_1", "workshop_2": "workshop_2"},
    )
    builder.add_edge("workshop_2", "workshop_3")
    builder.add_edge("workshop_3", "workshop_4")
    builder.add_conditional_edges(
        "workshop_4",
        route_after_arbitration,
        {"reset_feedback": "reset_feedback", END: END},
    )
    builder.add_edge("reset_feedback", "workshop_1")

    return builder.compile()


# ═══════════════════════════════════════════════════════════════════════════════
#  节点标签映射
# ═══════════════════════════════════════════════════════════════════════════════

NODE_LABELS = {
    "workshop_1":    "🏛️ 车间1 · 架构师内阁",
    "workshop_2":    "📄 车间2 · 文档压制",
    "workshop_3":    "⚔️ 车间3 · 红蓝对抗",
    "workshop_4":    "⚖️ 车间4 · 全局仲裁",
    "reset_feedback": "🔄 重置反馈状态",
}


# ═══════════════════════════════════════════════════════════════════════════════
#  Streamlit 界面
#  UI 层只消费 display_messages（只追加，永不清空），不接触 round_messages
# ═══════════════════════════════════════════════════════════════════════════════

def _new_session(title: str) -> dict:
    """[feat2] 创建新话题数据结构。"""
    return {
        "title":           title,
        "chat_history":    [],
        "final_whitepaper": "",
        "summary_context": "",   # [feat3] 该话题的跨轮摘要
    }


def _auto_title(client: OpenAI, model: str, prompt: str) -> str:
    """[feat2] 根据首条需求自动生成简短话题标题，失败时静默返回空字符串。"""
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0.3,
            messages=[
                {"role": "system", "content":
                    "你是一个标签生成器，只输出不超过 10 个字的话题标签，不要标点符号和引号。"},
                {"role": "user", "content": f"需求：{prompt}"},
            ],
        )
        return resp.choices[0].message.content.strip()[:12]
    except Exception:
        return ""


def _export_markdown(chat_history: list, title: str) -> str:
    """[feat2] 导出当前话题对话历史为 Markdown。"""
    md  = f"# 🏗️ 白皮书工坊讨论记录 - {title}\n\n"
    md += f"> 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n\n---\n\n"
    for msg in chat_history:
        role_label = "👤 用户" if msg["role"] == "user" else "🤖 AI"
        md += f"**{role_label}**\n\n{msg['content']}\n\n---\n\n"
    return md


def main():
    st.set_page_config(page_title="多智能体白皮书工坊", page_icon="🏗️", layout="wide")
    st.title("🏗️ 多智能体白皮书工坊")
    st.caption("四车间流水线：架构师内阁 → 文档压制 → 红蓝对抗 → 全局仲裁")

    # ── [feat2] Session State 初始化 ─────────────────────────────────────────
    if "sessions" not in st.session_state:
        st.session_state.sessions = {"default": _new_session("默认话题")}
    if "current_sid" not in st.session_state:
        st.session_state.current_sid = "default"

    # ── 侧边栏 ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ API 配置")
        api_key = st.text_input("API Key", type="password", placeholder="sk-...")
        base_url = st.text_input(
            "Base URL",
            value="https://api.deepseek.com",
            help="支持任意 OpenAI 兼容端点",
        )
        model_name = st.text_input(
            "模型名称",
            value="deepseek-chat",
            help="deepseek-chat / gpt-4o / qwen-plus 等\n不支持 JSON 模式的模型会自动降级",
        )

        st.divider()

        # ── [feat4] 输出长度控制 ─────────────────────────────────────────────
        st.header("📏 输出长度")
        detail_level = st.radio(
            "架构师发言篇幅：",
            ["简洁", "标准", "详细"],
            index=1,
            help=(
                "简洁：≤150字，快速迭代，大幅节省 Token 和等待时间\n"
                "标准：模型自主判断\n"
                "详细：充分展开所有细节和设计权衡，适合最终白皮书生成"
            ),
        )
        detail_instruction = DETAIL_INSTRUCTIONS[detail_level]

        st.divider()

        # ── [feat2] 话题管理 ─────────────────────────────────────────────────
        st.header("💬 话题管理")

        col_new, col_del = st.columns([3, 1])
        with col_new:
            if st.button("➕ 新建话题", use_container_width=True):
                new_sid = f"sess_{int(time.time())}"
                st.session_state.sessions[new_sid] = _new_session(
                    f"话题_{time.strftime('%H%M%S')}"
                )
                st.session_state.current_sid = new_sid
                st.rerun()
        with col_del:
            if st.button("🗑️", help="删除当前话题", use_container_width=True):
                if len(st.session_state.sessions) > 1:
                    del st.session_state.sessions[st.session_state.current_sid]
                    st.session_state.current_sid = list(st.session_state.sessions.keys())[0]
                    st.rerun()
                else:
                    st.warning("至少保留一个话题。")

        sid_list   = list(st.session_state.sessions.keys())
        title_list = [st.session_state.sessions[s]["title"] for s in sid_list]
        cur_idx    = sid_list.index(st.session_state.current_sid)
        selected   = st.selectbox("切换话题", title_list, index=cur_idx)
        st.session_state.current_sid = sid_list[title_list.index(selected)]

        cur_sess = st.session_state.sessions[st.session_state.current_sid]

        # Markdown 导出
        if cur_sess["chat_history"]:
            st.download_button(
                label="📄 导出为 Markdown",
                data=_export_markdown(cur_sess["chat_history"], cur_sess["title"]),
                file_name=f"{cur_sess['title']}_{time.strftime('%Y%m%d')}.md",
                mime="text/markdown",
                use_container_width=True,
            )

        # [feat3] 摘要预览
        if cur_sess["summary_context"]:
            with st.expander("🧠 跨轮摘要记忆", expanded=False):
                st.caption("AI 自动压缩的历史摘要，用于跨轮长程记忆（防遗忘）：")
                st.markdown(cur_sess["summary_context"])

        st.divider()
        st.markdown(
            "**退火规则**\n"
            "- 1–3 轮：任何瑕疵打回\n"
            "- 4–6 轮：仅致命/严重打回\n"
            "- 7+ 轮：除致命外强制通过"
        )
        st.divider()
        if st.button("🗑️ 清空当前话题历史", use_container_width=True):
            cur_sess["chat_history"]     = []
            cur_sess["final_whitepaper"] = ""
            cur_sess["summary_context"]  = ""
            st.session_state.json_mode_unsupported = set()   # [fix5]
            st.rerun()

    # ── 主区域：渲染当前话题历史 ─────────────────────────────────────────────
    cur_sess = st.session_state.sessions[st.session_state.current_sid]

    for msg in cur_sess["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── 用户输入 ─────────────────────────────────────────────────────────────
    user_prompt = st.chat_input("请输入你的系统/产品需求描述…")

    if user_prompt:
        if not api_key:
            st.error("请先在侧边栏填写 API Key。")
            return

        with st.chat_message("user"):
            st.markdown(user_prompt)
        cur_sess["chat_history"].append({"role": "user", "content": user_prompt})

        client = OpenAI(api_key=api_key, base_url=base_url)

        # [feat2] 首次发言自动生成话题标题
        is_first = sum(1 for m in cur_sess["chat_history"] if m["role"] == "user") == 1
        if is_first and (cur_sess["title"].startswith("话题_") or cur_sess["title"] == "默认话题"):
            new_title = _auto_title(client, model_name, user_prompt)
            if new_title:
                cur_sess["title"] = new_title

        graph = build_graph(client, model_name, detail_instruction)

        initial_state: AgentState = {
            "round_messages":    [],
            "display_messages":  [],
            "whitepaper":        "",
            "feedback_fatal":    "",
            "feedback_minor":    "",
            "loop_count":        0,
            "consensus_reached": False,
            "final_decision":    "",
            "user_prompt":       user_prompt,
            "arbitration_reason": "",
            "cabinet_call_count": 0,
            "summary_context":   cur_sess["summary_context"],   # [feat3] 注入已有摘要
        }

        status_bar = st.empty()
        latest_whitepaper = ""

        try:
            for event in graph.stream(initial_state, stream_mode="updates"):
                node_name   = list(event.keys())[0]
                node_output = event[node_name]

                label = NODE_LABELS.get(node_name, node_name)
                status_bar.info(f"⚙️ 当前车间: {label}")

                # [feat3] 节点更新摘要时同步到 session
                if node_output.get("summary_context"):
                    cur_sess["summary_context"] = node_output["summary_context"]

                # UI 消费 display_messages 增量
                # [feat1] __streamed__ 消息已在节点内流式直接渲染，跳过重复渲染，但仍记录 history
                for m in node_output.get("display_messages", []):
                    content = m.get("content", "")
                    if not content:
                        continue
                    if m.get("__streamed__"):
                        # 已流式渲染，只追加到 history 供历史回放（带标记）
                        cur_sess["chat_history"].append(m)
                        continue
                    with st.chat_message("assistant"):
                        st.markdown(content)
                    cur_sess["chat_history"].append({"role": "assistant", "content": content})

                if node_output.get("whitepaper"):
                    latest_whitepaper = node_output["whitepaper"]

                if node_output.get("final_decision") == "汇报老板":
                    status_bar.success("✅ 白皮书已通过全局仲裁，推演完成！")

        except Exception as e:
            st.error(f"运行出错：{e}")
            status_bar.error("❌ 流水线中断")

        if latest_whitepaper:
            cur_sess["final_whitepaper"] = latest_whitepaper
            st.divider()
            st.subheader("📋 最终白皮书")
            with st.expander("点击展开查看完整白皮书", expanded=True):
                st.markdown(latest_whitepaper)
            st.download_button(
                label="⬇️ 下载白皮书 (Markdown)",
                data=latest_whitepaper.encode("utf-8"),
                file_name=f"{cur_sess['title']}_whitepaper.md",
                mime="text/markdown",
            )

    elif cur_sess["final_whitepaper"]:
        st.divider()
        st.subheader("📋 最终白皮书")
        with st.expander("点击展开查看完整白皮书", expanded=False):
            st.markdown(cur_sess["final_whitepaper"])
        st.download_button(
            label="⬇️ 下载白皮书 (Markdown)",
            data=cur_sess["final_whitepaper"].encode("utf-8"),
            file_name=f"{cur_sess['title']}_whitepaper.md",
            mime="text/markdown",
        )


if __name__ == "__main__":
    main()