"""
多智能体白皮书工坊 (Multi-Agent Whitepaper Workshop) — v4 · 统一版
=====================================================================
技术栈: Python 3.10+, LangGraph >=0.2.0, Streamlit, OpenAI SDK (兼容 DeepSeek/OpenAI)

🆕 统一版新增：
  ✅ [unified] 双模式切换 : 侧边栏一键切换「通用智囊团」/「程序架构师」两套 Prompt
              切换即时生效，下次发言自动使用新模式；话题历史按模式独立保存

四车间流水线:
  1. 架构师内阁 — 主持人动态点名(图驱动单步) + 三位架构师脑暴/修复
  2. 文档压制   — 将讨论凝炼为结构化白皮书 JSON → Markdown
  3. 红蓝对抗   — 挑刺师1(致命 Fail-Fast) + 挑刺师2(次要)
  4. 全局仲裁   — 退火策略决定 "打回重做" 或 "汇报老板"

运行方式:
  pip install langgraph streamlit openai
  streamlit run workshop_unified.py
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
# ═══════════════════════════════════════════════════════════════════════════════

def smart_add(left: List[dict], right: List[dict]) -> List[dict]:
    if any(m.get("__clear__") for m in right):
        remainder = [m for m in right if not m.get("__clear__")]
        return remainder
    return left + right


# ═══════════════════════════════════════════════════════════════════════════════
#  AgentState
# ═══════════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    round_messages:   Annotated[List[dict], smart_add]
    display_messages: Annotated[List[dict], operator.add]
    whitepaper:        str
    feedback_fatal:    str
    feedback_minor:    str
    loop_count:        int
    consensus_reached: bool
    final_decision:    str
    user_prompt:       str
    arbitration_reason: str
    cabinet_call_count: int
    summary_context:   str


# ═══════════════════════════════════════════════════════════════════════════════
#  Token 估算 & 截断工具
# ═══════════════════════════════════════════════════════════════════════════════

_RE_CJK = re.compile(r'[一-鿿㐀-䶿豈-﫿]')

def _estimate_tokens(text: str) -> int:
    if not text:
        return 1
    cjk_count   = len(_RE_CJK.findall(text))
    other_count = len(text) - cjk_count
    return max(1, int(cjk_count * 1.0 + other_count * 0.3))


def trim_by_token_budget(messages: List[dict], token_budget: int = 3000) -> List[dict]:
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
    trimmed = trim_by_token_budget(messages, token_budget)
    return "\n".join(
        f"[{m.get('role', '?')}] {m.get('content', '')}" for m in trimmed
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  [unified] 双模式 Prompt 定义
#
#  MODE_GENERAL  — 通用智囊团（小说/产品/生活计划等任意场景）
#  MODE_ENGINEER — 程序架构师（软件/技术架构专用）
# ═══════════════════════════════════════════════════════════════════════════════

MODE_GENERAL  = "通用智囊团"
MODE_ENGINEER = "程序架构师"

PROMPTS = {
    MODE_GENERAL: {
        "moderator": textwrap.dedent("""\
            你是智囊团的最高决策主持人。
            你的职责是根据当前讨论进展，动态决定下一步行动。
            只输出 JSON，格式：
            {"action": "call"|"consensus", "target": "专家A"|"专家B"|"专家C"|null, "reason": "简要理由"}
            - action=call 表示点名某位专家发言（每轮最多点名 3 次后必须 consensus）。
            - action=consensus 表示多维度的讨论已充分，可以进入方案总结。
        """),
        "agents": {
            "专家A": textwrap.dedent("""\
                你是"核心逻辑与战略专家"（专家A）。
                针对当前议题，你需要侧重于：第一性原理、核心主线、主要情节（若为小说）或最关键的成功要素。
                请根据当前讨论上下文给出你的专业意见。
                直接输出意见文本，不要 JSON 格式，不要代码块包裹。
            """),
            "专家B": textwrap.dedent("""\
                你是"资源与可行性专家"（专家B）。
                针对当前议题，你需要侧重于：现实约束、资源调配、背景设定（若为小说/游戏）或具体可落地的执行细节。
                请根据当前讨论上下文给出你的专业意见。
                直接输出意见文本，不要 JSON 格式，不要代码块包裹。
            """),
            "专家C": textwrap.dedent("""\
                你是"风险与体验专家"（专家C）。
                针对当前议题，你需要侧重于：外部交互、用户/读者体验、潜在的阻力以及边缘情况（如小概率失败场景）。
                请根据当前讨论上下文给出你的专业意见。
                直接输出意见文本，不要 JSON 格式，不要代码块包裹。
            """),
        },
        "doc_compress": textwrap.dedent("""\
            你是高级方案提炼专家。
            请将智囊团的讨论精华提炼为一份结构化的《终极推演方案》。
            由于系统限制，你必须严格使用以下四个固定 JSON 键值来输出内容，请根据当前议题（如写小说、做产品、生活计划等）灵活变通其含义：

            严格输出 JSON，格式：
            {
                "core_flow": "用于描述：核心主线、情节大纲、或最关键的行动路径",
                "data_structure": "用于描述：人物设定、世界观、资源盘点、或底层逻辑支持",
                "api_definition": "用于描述：关键里程碑、外部交互动作、章节目录、或阶段性指标",
                "rejected_ideas": "用于描述：被否决的想法、避坑指南、或不可行的假设"
            }
        """),
        "red_fatal": textwrap.dedent("""\
            你是红队挑刺师1号（致命缺陷检测）。
            审查方案，找出致命级别问题（如：极其反常理的逻辑、导致全盘崩溃的现实阻碍、严重的人设矛盾、或是绝无可能实现的空想）。
            输出 JSON：{"fatal_issues": "致命问题描述，没有则为空字符串 ''"}
        """),
        "blue_minor": textwrap.dedent("""\
            你是蓝队挑刺师2号（次要瑕疵检测）。
            审查方案，找出次要问题（如：细节不够丰满、执行效率低、部分设定略显俗套或冗余等）。
            输出 JSON：{"minor_issues": "次要问题描述，没有则为空字符串 ''"}
        """),
        "summarizer": textwrap.dedent("""\
            你是高效的会议纪要员。
            请将以下多维度推演讨论高度浓缩，提取：
            1. 议题的核心目标
            2. 已确认的关键路径或设定
            3. 被明确否决的方案或踩坑点
            输出纯文本，控制在 300 字以内，作为后续讨论的背景记忆。
        """),
        "detail_instructions": {
            "简洁": "\n\n【输出长度要求】请控制在 500 字以内，只给出核心结论，省略推导过程。",
            "标准": "",
            "详细": "\n\n【输出长度要求】请充分展开，覆盖所有相关细节、边界情况和设计权衡，不限字数。",
        },
        "agent_icon": "🧠",
        "doc_section": {
            "core_flow": "## 核心主线 / 行动路径",
            "data_structure": "## 人物/世界/资源设定",
            "api_definition": "## 里程碑 / 关键节点",
            "rejected_ideas": "## 被否决方案 / 避坑指南",
        },
        "doc_title": "# 推演方案白皮书",
    },

    MODE_ENGINEER: {
        "moderator": textwrap.dedent("""\
            你是架构师内阁的最高决策主持人。
            你的职责是根据当前讨论进展，动态决定下一步行动。
            只输出 JSON，格式：
            {"action": "call"|"consensus", "target": "架构师A"|"架构师B"|"架构师C"|null, "reason": "简要理由"}
            - action=call 表示点名某位架构师发言（每轮最多点名 3 次后必须 consensus）。
            - action=consensus 表示讨论已充分，可以进入文档压制。
        """),
        "agents": {
            "架构师A": textwrap.dedent("""\
                你是主业务架构师（架构师A），专注核心业务逻辑与领域建模。
                请根据当前讨论上下文给出你的专业意见。
                直接输出意见文本，不要 JSON 格式，不要代码块包裹。
            """),
            "架构师B": textwrap.dedent("""\
                你是数据/后端架构师（架构师B），专注数据库设计、API 接口、性能优化。
                请根据当前讨论上下文给出你的专业意见。
                直接输出意见文本，不要 JSON 格式，不要代码块包裹。
            """),
            "架构师C": textwrap.dedent("""\
                你是前端/交互架构师（架构师C），专注用户体验、界面流程与前端架构。
                请根据当前讨论上下文给出你的专业意见。
                直接输出意见文本，不要 JSON 格式，不要代码块包裹。
            """),
        },
        "doc_compress": textwrap.dedent("""\
            你是高级技术文档压制专家。
            将架构师们的讨论精华提炼为结构化技术白皮书。
            严格输出 JSON，格式：
            {
                "core_flow": "核心业务流程描述",
                "data_structure": "数据结构与模型设计",
                "api_definition": "API 接口定义",
                "rejected_ideas": "被否决方案及原因"
            }
        """),
        "red_fatal": textwrap.dedent("""\
            你是红队挑刺师1号（致命缺陷检测）。
            审查白皮书，找出致命级别问题（安全漏洞、逻辑矛盾、数据丢失风险、不可行方案等）。
            输出 JSON：{"fatal_issues": "致命问题描述，没有则为空字符串 ''"}
        """),
        "blue_minor": textwrap.dedent("""\
            你是蓝队挑刺师2号（次要瑕疵检测）。
            审查白皮书，找出次要问题（命名不一致、文档遗漏、可优化点等）。
            输出 JSON：{"minor_issues": "次要问题描述，没有则为空字符串 ''"}
        """),
        "summarizer": textwrap.dedent("""\
            你是高效的会议纪要员。
            请将以下架构讨论高度浓缩，提取：
            1. 用户的核心需求要点
            2. 已确认的架构决策
            3. 被明确否决的方案
            输出纯文本，控制在 300 字以内，作为后续讨论的背景记忆。
        """),
        "detail_instructions": {
            "简洁": "\n\n【输出长度要求】请控制在 150 字以内，只给出核心结论，省略推导过程。",
            "标准": "",
            "详细": "\n\n【输出长度要求】请充分展开，覆盖所有相关细节、边界情况和设计权衡，不限字数。",
        },
        "agent_icon": "🏗️",
        "doc_section": {
            "core_flow": "## 核心业务流程",
            "data_structure": "## 数据结构与模型",
            "api_definition": "## API 接口定义",
            "rejected_ideas": "## 被否决方案",
        },
        "doc_title": "# 技术白皮书",
    },
}

SYS_ARBITRATOR = textwrap.dedent("""\
    你是全局仲裁官，掌握最终决策权。
    根据红蓝对抗反馈和当前迭代轮次，决定当前推演方案是否足够严谨、丰满且可以直接交付。
    输出 JSON：{"decision": "打回重做"|"汇报老板", "reason": "决策理由"}

    退火规则（必须严格遵守）：
    - 第 1-3 轮：任何瑕疵（致命或次要）都应打回重做
    - 第 4-6 轮：仅致命或严重现实阻碍才打回，次要问题可放行
    - 第 7 轮及以上：除非存在彻底不可行的致命缺陷，否则强制通过（汇报老板）
""")


# ═══════════════════════════════════════════════════════════════════════════════
#  LLM 调用工具
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_json_from_text(text: str) -> dict | None:
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
    if "JSON" not in system.upper():
        system += "\n请以 JSON 格式输出，不要输出任何其他内容。"

    messages_payload = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
    raw = ""

    try:
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
                else:
                    raise

        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=messages_payload,
        )
        raw = resp.choices[0].message.content.strip()

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


def call_llm_stream(
    client: OpenAI,
    model: str,
    system: str,
    user_content: str,
    placeholder,
    temperature: float = 0.7,
) -> str:
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
                placeholder.markdown(full_text + "▌")
        placeholder.markdown(full_text)
    except Exception as e:
        full_text = f"⚠️ 流式调用失败: {e}"
        placeholder.markdown(full_text)
    return full_text


def call_llm_summary(client: OpenAI, model: str, text: str, summarizer_prompt: str) -> str:
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0.3,
            messages=[
                {"role": "system", "content": summarizer_prompt},
                {"role": "user",   "content": text},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
#  节点工厂（闭包注入 client, model, mode_prompts, detail_instruction）
# ═══════════════════════════════════════════════════════════════════════════════

def make_nodes(
    client: OpenAI,
    model: str,
    mode_prompts: dict,
    detail_instruction: str = "",
) -> dict:

    CABINET_CALL_LIMIT = 5
    agent_prompts = mode_prompts["agents"]
    agent_icon    = mode_prompts["agent_icon"]

    def _inject_summary(base_prompt: str, summary: str) -> str:
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
        call_count  = state.get("cabinet_call_count", 0)
        summary     = state.get("summary_context", "")

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
                "cabinet_call_count": 0,
            }

        recent_ctx = fmt_msgs(state.get("round_messages", []), token_budget=2500)
        moderator_sys = _inject_summary(mode_prompts["moderator"], summary)

        if loop_count == 0:
            moderator_input = (
                f"【脑暴模式】用户原始需求：\n{user_prompt}\n\n"
                f"本轮历史对话（已进行 {call_count} 次发言）：\n{recent_ctx}\n\n"
                f"剩余可点名次数：{CABINET_CALL_LIMIT - call_count} 次。"
                "请决定点名某位专家/架构师发言，或判断讨论已充分可达成共识（consensus）。"
            )
        else:
            moderator_input = (
                f"【修复模式 · 第 {loop_count} 轮返工】\n"
                f"用户需求：{user_prompt}\n"
                f"致命问题：{fatal or '无'}\n次要问题：{minor or '无'}\n"
                f"白皮书摘要：\n{whitepaper[:600]}\n\n"
                f"本轮历史对话（已进行 {call_count} 次发言）：\n{recent_ctx}\n\n"
                f"剩余可点名次数：{CABINET_CALL_LIMIT - call_count} 次。"
                "请点名局部修复，或判断已修复完毕达成共识（consensus）。"
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
                "cabinet_call_count": 0,
            }

        arch_base_sys = agent_prompts.get(target, list(agent_prompts.values())[0])
        arch_sys = _inject_summary(arch_base_sys, summary) + detail_instruction

        if loop_count == 0:
            arch_input = (
                f"用户需求：{user_prompt}\n\n"
                f"本轮历史对话：\n{recent_ctx}\n\n"
                "请基于上述背景提出你的专业意见。"
            )
        else:
            arch_input = (
                f"用户需求：{user_prompt}\n"
                f"【警告】上一版被打回，请仅针对以下痛点局部修复：\n"
                f"致命反馈：{fatal or '无'}\n次要反馈：{minor or '无'}\n"
                f"现有方案：\n{whitepaper}\n\n"
                f"本轮历史对话：\n{recent_ctx}"
            )

        host_msg = {"role": "assistant", "content": f"🎙️ **【主持人】** 呼叫 {target}。理由：{reason}"}

        with st.chat_message("assistant"):
            st.markdown(host_msg["content"])
            st.markdown(f"{agent_icon} **【{target}】** 正在发言...")
            arch_placeholder = st.empty()

        opinion = call_llm_stream(client, model, arch_sys, arch_input, arch_placeholder)

        arch_msg = {"role": "assistant", "content": f"{agent_icon} **【{target}】** {opinion}"}
        arch_msg_display = {**arch_msg, "__streamed__": True}

        return {
            "round_messages":    [host_msg, arch_msg],
            "display_messages":  [host_msg, arch_msg_display],
            "consensus_reached": False,
            "cabinet_call_count": call_count + 1,
        }

    def workshop_2_doc_compression(state: AgentState) -> dict:
        user_prompt  = state.get("user_prompt", "")
        round_msgs   = state.get("round_messages", [])
        prev_summary = state.get("summary_context", "")

        new_summary = ""
        if round_msgs:
            text_to_compress = "\n".join(
                f"[{m.get('role','?')}] {m.get('content','')}" for m in round_msgs
            )
            new_summary = call_llm_summary(client, model, text_to_compress, mode_prompts["summarizer"])

        if new_summary:
            if prev_summary:
                summary_blocks = prev_summary.split("\n\n---（新一轮）---\n\n")
                summary_blocks.append(new_summary)
                if len(summary_blocks) > 2:
                    summary_blocks = summary_blocks[-2:]
                summary_context = "\n\n---（新一轮）---\n\n".join(summary_blocks)
            else:
                summary_context = new_summary
        else:
            summary_context = prev_summary

        discussion = fmt_msgs(round_msgs, token_budget=4000)
        user_input = (
            f"用户原始需求：\n{user_prompt}\n\n"
            f"讨论记录：\n{discussion}\n\n"
            "请提炼为结构化方案。"
        )
        result = call_llm_json(client, model, mode_prompts["doc_compress"], user_input, temperature=0.3)

        # [fix-dynamic-doc] 完全动态化生成文档，彻底解耦 doc_section。
        # 新增/删除/重排字段只需修改 PROMPTS 里的 doc_section，核心函数无需改动。
        sec = mode_prompts["doc_section"]
        wp_md = f"{mode_prompts['doc_title']}\n\n"
        for key, section_title in sec.items():
            default = "（无）" if key == "rejected_ideas" else "（未生成）"
            content = result.get(key, default)
            wp_md += f"{section_title}\n{content}\n\n"

        notice_msg = {"role": "assistant", "content": "📄 **白皮书已生成/更新**（详见下方展开区域）"}

        return {
            "round_messages":   [{"__clear__": True}],
            "display_messages": [notice_msg],
            "whitepaper":       wp_md,
            "summary_context":  summary_context,
        }

    def workshop_3_red_blue_test(state: AgentState) -> dict:
        whitepaper = state.get("whitepaper", "")

        red_result = call_llm_json(
            client, model, mode_prompts["red_fatal"],
            f"请审查以下方案的致命缺陷：\n\n{whitepaper}",
            temperature=0.2,
        )
        fatal = red_result.get("fatal_issues", "").strip()

        if fatal:
            msg = {"role": "assistant", "content": f"🔴 **挑刺师1（致命）**：{fatal}\n⚡ Fail-Fast，跳过次要检测。"}
            return {"feedback_fatal": fatal, "feedback_minor": "", "display_messages": [msg]}

        blue_result = call_llm_json(
            client, model, mode_prompts["blue_minor"],
            f"请审查以下方案的次要问题：\n\n{whitepaper}",
            temperature=0.2,
        )
        minor = blue_result.get("minor_issues", "").strip()

        msgs = [
            {"role": "assistant", "content": "🔴 **挑刺师1（致命）**：未发现致命缺陷 ✅"},
            {"role": "assistant", "content": f"🔵 **挑刺师2（次要）**：{minor if minor else '未发现次要问题 ✅'}"},
        ]
        return {"feedback_fatal": "", "feedback_minor": minor, "display_messages": msgs}

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

    def reset_feedback_node(state: AgentState) -> dict:
        return {
            "feedback_fatal":    "",
            "feedback_minor":    "",
            "consensus_reached": False,
            "cabinet_call_count": 0,
            "round_messages":    [{"__clear__": True}],
        }

    return {
        "workshop_1":    workshop_1_architect_cabinet,
        "workshop_2":    workshop_2_doc_compression,
        "workshop_3":    workshop_3_red_blue_test,
        "workshop_4":    workshop_4_global_arbitration,
        "reset_feedback": reset_feedback_node,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  路由 & 图构建
# ═══════════════════════════════════════════════════════════════════════════════

def route_after_cabinet(state: AgentState) -> str:
    return "workshop_2" if state.get("consensus_reached") else "workshop_1"

def route_after_arbitration(state: AgentState) -> str:
    return END if state.get("final_decision") == "汇报老板" else "reset_feedback"

def build_graph(client: OpenAI, model: str, mode_prompts: dict, detail_instruction: str = "") -> CompiledStateGraph:
    nodes = make_nodes(client, model, mode_prompts, detail_instruction)
    builder = StateGraph(AgentState)
    for name, func in nodes.items():
        builder.add_node(name, func)
    builder.add_edge(START, "workshop_1")
    builder.add_conditional_edges(
        "workshop_1", route_after_cabinet,
        {"workshop_1": "workshop_1", "workshop_2": "workshop_2"},
    )
    builder.add_edge("workshop_2", "workshop_3")
    builder.add_edge("workshop_3", "workshop_4")
    builder.add_conditional_edges(
        "workshop_4", route_after_arbitration,
        {"reset_feedback": "reset_feedback", END: END},
    )
    builder.add_edge("reset_feedback", "workshop_1")
    return builder.compile()

NODE_LABELS = {
    "workshop_1":    "🏛️ 车间1 · 内阁议事",
    "workshop_2":    "📄 车间2 · 文档压制",
    "workshop_3":    "⚔️ 车间3 · 红蓝对抗",
    "workshop_4":    "⚖️ 车间4 · 全局仲裁",
    "reset_feedback": "🔄 重置反馈状态",
}


# ═══════════════════════════════════════════════════════════════════════════════
#  Streamlit UI 辅助函数
# ═══════════════════════════════════════════════════════════════════════════════

def _new_session(title: str) -> dict:
    return {
        "title":           title,
        "chat_history":    [],
        "final_whitepaper": "",
        "summary_context": "",
    }

def _auto_title(client: OpenAI, model: str, prompt: str) -> str:
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
    md  = f"# 🏗️ 白皮书工坊讨论记录 - {title}\n\n"
    md += f"> 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n\n---\n\n"
    for msg in chat_history:
        role_label = "👤 用户" if msg["role"] == "user" else "🤖 AI"
        md += f"**{role_label}**\n\n{msg['content']}\n\n---\n\n"
    return md


# ═══════════════════════════════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(page_title="多智能体白皮书工坊", page_icon="🏗️", layout="wide")

    # ── Session State 初始化 ──────────────────────────────────────────────────
    if "sessions" not in st.session_state:
        st.session_state.sessions = {"default": _new_session("默认话题")}
    if "current_sid" not in st.session_state:
        st.session_state.current_sid = "default"
    if "current_mode" not in st.session_state:
        st.session_state.current_mode = MODE_ENGINEER

    # ── 侧边栏 ────────────────────────────────────────────────────────────────
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
            help="deepseek-chat / gpt-4o / qwen-plus 等",
        )

        st.divider()

        # ── [unified] 模式切换（核心新增）────────────────────────────────────
        st.header("🔀 专家团模式")
        
        prev_mode = st.session_state.current_mode
        mode_choice = st.radio(
            "选择 Prompt 套件：",
            [MODE_ENGINEER, MODE_GENERAL],
            index=0 if st.session_state.current_mode == MODE_ENGINEER else 1,
            help=(
                f"**{MODE_ENGINEER}**：三位架构师（业务/后端/前端），适合软件开发、系统设计\n\n"
                f"**{MODE_GENERAL}**：三位专家（战略/可行性/风险），适合小说策划、产品规划、生活决策等"
            ),
        )
        st.session_state.current_mode = mode_choice

        # ── [fix-clash] 跨模式上下文污染拦截 ───────────────────────────────────
        # 如果当前话题已有历史且发生了模式切换，必须处理上下文污染风险。
        # 提供两条路径：①新建干净话题继续切换；②撤销切换留在原模式。
        if prev_mode != mode_choice:
            _cur_hist = st.session_state.sessions[st.session_state.current_sid]["chat_history"]
            if _cur_hist:
                # 有历史 → 拦截，给用户选择
                st.warning(
                    f"⚠️ **切换模式可能导致上下文污染**\n\n"
                    f"当前话题已有 {len(_cur_hist)} 条历史（{prev_mode}模式），"
                    f"直接切换到「{mode_choice}」会把旧上下文带入新 Prompt，"
                    f"干扰模型注意力。\n\n"
                    f"请选择处理方式："
                )
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("➕ 新建话题并切换", use_container_width=True, type="primary"):
                        new_sid = f"sess_{int(time.time())}"
                        st.session_state.sessions[new_sid] = _new_session(
                            f"话题_{time.strftime('%H%M%S')}"
                        )
                        st.session_state.current_sid = new_sid
                        # 模式已被 radio 写入 session_state，保持新模式
                        st.rerun()
                with col_b:
                    if st.button("↩️ 撤销切换", use_container_width=True):
                        # 回滚到切换前的模式
                        st.session_state.current_mode = prev_mode
                        st.rerun()
                # 阻止后续代码使用新模式执行（本次渲染仍用旧模式）
                mode_choice = prev_mode
            else:
                # 无历史（空话题）→ 无风险，直接切换
                st.success(f"✅ 已切换到「{mode_choice}」模式")

        # 当前模式的专家说明
        cur_agents = PROMPTS[mode_choice]["agents"]
        with st.expander("📋 当前专家团成员", expanded=False):
            for name, prompt in cur_agents.items():
                first_line = prompt.strip().split("\n")[0].replace("你是", "").strip()
                st.markdown(f"- **{name}**：{first_line}")

        st.divider()

        # ── 输出长度控制 ──────────────────────────────────────────────────────
        st.header("📏 输出长度")
        detail_level = st.radio(
            "发言篇幅：",
            ["简洁", "标准", "详细"],
            index=1,
            help=(
                "简洁：快速迭代，节省 Token\n"
                "标准：模型自主判断\n"
                "详细：充分展开所有细节"
            ),
        )
        detail_instruction = PROMPTS[mode_choice]["detail_instructions"][detail_level]

        st.divider()

        # ── 话题管理 ──────────────────────────────────────────────────────────
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

        if cur_sess["chat_history"]:
            st.download_button(
                label="📄 导出为 Markdown",
                data=_export_markdown(cur_sess["chat_history"], cur_sess["title"]),
                file_name=f"{cur_sess['title']}_{time.strftime('%Y%m%d')}.md",
                mime="text/markdown",
                use_container_width=True,
            )

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
            st.session_state.json_mode_unsupported = set()
            st.rerun()

    # ── 主区域标题（显示当前模式）────────────────────────────────────────────
    mode_icon = "🏗️" if mode_choice == MODE_ENGINEER else "🧠"
    st.title(f"{mode_icon} 多智能体白皮书工坊")
    
    col_title, col_badge = st.columns([5, 1])
    with col_title:
        st.caption("四车间流水线：内阁议事 → 文档压制 → 红蓝对抗 → 全局仲裁")
    with col_badge:
        badge_color = "#1f6feb" if mode_choice == MODE_ENGINEER else "#7c4dff"
        st.markdown(
            f'<span style="background:{badge_color};color:white;padding:3px 10px;'
            f'border-radius:12px;font-size:0.75rem;font-weight:600">{mode_choice}</span>',
            unsafe_allow_html=True,
        )

    # ── 渲染当前话题历史 ──────────────────────────────────────────────────────
    cur_sess = st.session_state.sessions[st.session_state.current_sid]

    for msg in cur_sess["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── 用户输入 ──────────────────────────────────────────────────────────────
    mode_prompts = PROMPTS[mode_choice]
    placeholder_text = (
        "请输入你的系统/产品需求描述…"
        if mode_choice == MODE_ENGINEER
        else "请输入你的议题（小说方向、产品规划、生活决策等）…"
    )
    user_prompt = st.chat_input(placeholder_text)

    if user_prompt:
        if not api_key:
            st.error("请先在侧边栏填写 API Key。")
            return

        with st.chat_message("user"):
            st.markdown(user_prompt)
        cur_sess["chat_history"].append({"role": "user", "content": user_prompt})

        client = OpenAI(api_key=api_key, base_url=base_url)

        is_first = sum(1 for m in cur_sess["chat_history"] if m["role"] == "user") == 1
        if is_first and (cur_sess["title"].startswith("话题_") or cur_sess["title"] == "默认话题"):
            new_title = _auto_title(client, model_name, user_prompt)
            if new_title:
                cur_sess["title"] = new_title

        graph = build_graph(client, model_name, mode_prompts, detail_instruction)

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
            "summary_context":   cur_sess["summary_context"],
        }

        status_bar = st.empty()
        latest_whitepaper = ""

        try:
            for event in graph.stream(initial_state, stream_mode="updates"):
                node_name   = list(event.keys())[0]
                node_output = event[node_name]

                label = NODE_LABELS.get(node_name, node_name)
                status_bar.info(f"⚙️ 当前车间: {label}")

                if node_output.get("summary_context"):
                    cur_sess["summary_context"] = node_output["summary_context"]

                for m in node_output.get("display_messages", []):
                    content = m.get("content", "")
                    if not content:
                        continue
                    if m.get("__streamed__"):
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
            st.subheader("📋 最终方案白皮书")
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
        st.subheader("📋 最终方案白皮书")
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