import streamlit as st
from openai import OpenAI
import time

# ==========================================
# 1. 页面配置与核心数据结构初始化
# ==========================================
st.set_page_config(page_title="AI 虚拟董事会 Pro Max", page_icon="🏛️", layout="wide")
st.title("🏛️ AI 虚拟董事会 Pro Max")
st.caption("全自动连续对话流 | 无感上下文压缩 | 动态严格度退火")

if "sessions" not in st.session_state:
    st.session_state.sessions = {
        "default_0": {
            "title": "新讨论_默认",
            "messages": [],           
            "summary_context": "",    
            "last_summarized_idx": 0, 
            "user_turn_count": 0      
        }
    }
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = "default_0"
if "api_configs" not in st.session_state:
    st.session_state.api_configs = {
        "DeepSeek": {"key": "", "url": "https://api.deepseek.com", "model": "deepseek-chat"}
    }

# ==========================================
# 辅助函数
# ==========================================
def call_llm_sync(prompt, system_prompt="你是一个后台辅助 AI。"):
    config = st.session_state.api_configs["DeepSeek"]
    if not config["key"]: return ""
    try:
        client = OpenAI(api_key=config["key"], base_url=config["url"])
        resp = client.chat.completions.create(
            model=config["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return ""

def generate_markdown(messages, title):
    md_content = f"# 🏛️ AI 虚拟董事会讨论记录 - {title}\n\n"
    md_content += f"> 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n\n---\n\n"
    for msg in messages:
        if msg["role"] != "system":
            md_content += f"### 【{msg.get('name', '未知')}】\n{msg['content']}\n\n---\n\n"
    return md_content

# ==========================================
# 2. 侧边栏：核心控制台
# ==========================================
with st.sidebar:
    st.header("⚙️ 全局配置")
    
    with st.expander("🔑 模型与 API 配置", expanded=True):
        with st.form("api_config_form"):
            for name, config in st.session_state.api_configs.items():
                st.subheader(name)
                new_key = st.text_input(f"{name} Key", value=config["key"], type="password")
                new_url = st.text_input(f"{name} URL", value=config["url"])
                new_model = st.text_input(f"{name} Model", value=config["model"])
            
            submit_api = st.form_submit_button("✅ 提交 / 保存配置")
            if submit_api:
                st.session_state.api_configs[name]["key"] = new_key
                st.session_state.api_configs[name]["url"] = new_url
                st.session_state.api_configs[name]["model"] = new_model
                st.success("API 配置已生效！")

    st.divider()

    st.header("🚀 项目执行阶段")
    project_phase = st.radio(
        "当前重点：",
        [
            "🚩 架构与逻辑推演（严禁代码）", 
            "📄 细化功能与稳健性分析", 
            "📝 生成技术白皮书框架", 
            "💻 实战编码阶段（释放火力）"
        ],
        help="切换阶段会改变 AI 的行为模式。在逻辑跑通前，建议保持在第一阶段。"
    )

    st.divider()
    st.header("📏 输出详细程度")
    detail_level = st.radio(
        "控制 AI 回答的篇幅：",
        ["简洁", "标准", "详细"],
        index=1,
        help="简洁：抓住要点，避免展开；标准：AI 自主判断；详细：允许详细阐述。"
    )

    st.divider()

    st.header("💬 讨论话题")
    if st.button("➕ 开启新话题"):
        new_id = f"session_{int(time.time())}"
        st.session_state.sessions[new_id] = {
            "title": f"新讨论_{time.strftime('%H%M')}",
            "messages": [],
            "summary_context": "",
            "last_summarized_idx": 0,
            "user_turn_count": 0
        }
        st.session_state.current_session_id = new_id
        st.rerun()
    
    session_options = list(st.session_state.sessions.keys())
    session_titles = [st.session_state.sessions[sid]["title"] for sid in session_options]
    
    selected_title = st.selectbox(
        "切换历史话题", 
        session_titles, 
        index=session_options.index(st.session_state.current_session_id)
    )
    st.session_state.current_session_id = session_options[session_titles.index(selected_title)]

    current_msgs = st.session_state.sessions[st.session_state.current_session_id]["messages"]
    current_title = st.session_state.sessions[st.session_state.current_session_id]["title"]
    if current_msgs:
        st.download_button(
            label="📄 一键导出为 Markdown",
            data=generate_markdown(current_msgs, current_title),
            file_name=f"{current_title}_{time.strftime('%Y%m%d')}.md",
            mime="text/markdown"
        )

# ==========================================
# 3. 强化版角色设定
# ==========================================
ROLES = {
    "架构师": "你是技术架构师。负责提出初步的落地逻辑或架构方案。如果挑刺者指出了问题，请针对性地修正并给出升级方案。",
    
    "挑刺者": """你是严密的审计官。阅读架构师的方案，寻找逻辑漏洞、边界情况或性能隐患。
【输出指令】：你是否继续挑刺，取决于当前的系统指令。如果你决定不再反驳，请必须在回复开头明确输出：“【无重大异议】”，然后简单总结认可的原因。""",

    "主持人": """你是讨论的决策者。你需要阅读【架构师】和【挑刺者】的对话。
【核心任务】：判断当前的讨论是否已经把问题理清。如果挑刺者明确表示了“【无重大异议】”，你必须终止讨论并向老板汇报。
【输出规则】：必须严格遵循以下两种格式之一输出：
如果分歧仍在（挑刺者提出了新的问题），请输出：
【继续讨论】<在此说明为什么还需要讨论，并向架构师抛出需要补充的问题>
如果结论已清晰（挑刺者输出无重大异议，或问题已完美解决），可以向老板交差，请输出：
【汇报老板】<在此写下最终的总结性落地方案，语气要专业>"""
}

# ==========================================
# 4. 核心逻辑：UI 渲染与对话流
# ==========================================
current_session = st.session_state.sessions[st.session_state.current_session_id]

for msg in current_session["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(f"**【{msg.get('name', '老板')}】**:\n\n{msg['content']}")

def get_current_context(role_system_prompt):
    ctx = [{"role": "system", "content": role_system_prompt}]
    if current_session["summary_context"]:
        ctx.append({"role": "system", "content": f"【历史背景记忆】：\n{current_session['summary_context']}"})
    recent_msgs = current_session["messages"][current_session["last_summarized_idx"]:]
    for m in recent_msgs:
        r = "user" if m["role"] == "user" else "assistant"
        ctx.append({"role": r, "content": f"【{m.get('name')}】: {m['content']}"})
    return ctx

if prompt := st.chat_input("输入需求，团队将开始连续推演..."):
    if not st.session_state.api_configs["DeepSeek"]["key"]:
        st.error("请先在左侧保存 API Key！")
        st.stop()

    current_session["messages"].append({"role": "user", "name": "老板", "content": prompt})
    current_session["user_turn_count"] += 1
    with st.chat_message("user"):
        st.markdown(f"**【老板】**:\n\n{prompt}")

    if current_session["user_turn_count"] == 1 and current_session["title"].startswith("新讨论"):
        new_title = call_llm_sync(
            f"根据用户的首句话，提炼一个不超过8个字的标签。\n用户：{prompt}", 
            system_prompt="你是一个标签生成器，只输出标签文本，不要标点符号。"
        )
        if new_title:
            current_session["title"] = new_title[:10]

    if current_session["user_turn_count"] > 1 and current_session["user_turn_count"] % 3 == 0:
        msgs_to_compress = current_session["messages"][current_session["last_summarized_idx"]:]
        text_to_compress = "\n".join([f"{m.get('name')}: {m['content']}" for m in msgs_to_compress])
        compress_prompt = f"请高度浓缩以下历史对话，提取核心需求和已达成的共识，作为后续对话的背景记忆:\n{text_to_compress}"
        
        new_summary = call_llm_sync(compress_prompt)
        if new_summary:
            current_session["summary_context"] = new_summary
            current_session["last_summarized_idx"] = len(current_session["messages"])

    config = st.session_state.api_configs["DeepSeek"]
    client = OpenAI(api_key=config["key"], base_url=config["url"])
    
    MAX_LOOPS = 10 
    loop_count = 0
    conclusion_reached = False

    while not conclusion_reached and loop_count < MAX_LOOPS:
        loop_count += 1

        roles_in_round = ["架构师", "挑刺者", "主持人"]
        
        for role_name in roles_in_round:
            with st.chat_message("assistant"):
                st.write(f"**【{role_name}】** (第 {loop_count} 轮):")
                
                phase_instruction = ""
                if "严禁代码" in project_phase:
                    phase_instruction = "\n\n⚠️【强制执行】：当前处于架构推演阶段，禁止输出任何代码块。违者将扣除 Token 奖励。"
                elif "白皮书" in project_phase:
                    phase_instruction = "\n\n⚠️【任务】：请专注于文档结构和技术指标描述，不要进入具体的编码实现。"

                detail_instruction = ""
                if detail_level == "简洁":
                    detail_instruction = "\n\n【输出要求】：请尽量输出简洁的回答，抓住核心要点，避免冗长和细节展开。"
                elif detail_level == "详细":
                    detail_instruction = "\n\n【输出要求】：可以详细阐述，确保覆盖所有相关细节，但保持条理清晰。"

                dynamic_system_prompt = ROLES[role_name] + phase_instruction + detail_instruction
                
                # ==========================================
                # 【新增】核心逻辑：动态严格度退火
                # ==========================================
                if role_name == "挑刺者":
                    if loop_count <= 3:
                        dynamic_system_prompt += "\n\n🔥【当前阶段要求：极度苛刻】：现在是前3轮推演，你必须【吹毛求疵】！不要放过任何微小瑕疵、极端边缘场景或潜在的性能瓶颈。绝对不允许妥协，【严禁】在此阶段输出“无重大异议”。"
                    elif loop_count <= 6:
                        dynamic_system_prompt += "\n\n⚖️【当前阶段要求：抓大放小】：讨论已进入中期（第4-6轮），请适当放宽要求。重点关注核心逻辑的致命伤，忽略不影响大局的小瑕疵。如果核心架构已基本稳健，请直接输出【无重大异议】。"
                    else:
                        dynamic_system_prompt += "\n\n🤝【当前阶段要求：强制收敛】：讨论已进入后期（第7轮及以上）。为了推进项目落地，请极度务实。除非存在会导致系统立刻崩溃的致命错误，否则你必须妥协放行，直接输出【无重大异议】。"

                context = get_current_context(dynamic_system_prompt)
                
                full_response = ""
                message_placeholder = st.empty()
                
                try:
                    stream = client.chat.completions.create(
                        model=config["model"],
                        messages=context,
                        stream=True
                    )
                    
                    for chunk in stream:
                        if chunk.choices[0].delta.content is not None:
                            content = chunk.choices[0].delta.content
                            full_response += content
                            message_placeholder.markdown(full_response + "▌")
                    
                    message_placeholder.markdown(full_response)
                    
                    current_session["messages"].append({"role": "assistant", "name": role_name, "content": full_response})
                    
                    if role_name == "主持人":
                        if "【汇报老板】" in full_response:
                            conclusion_reached = True
                        elif "【继续讨论】" in full_response:
                            pass 
                        else:
                            conclusion_reached = True

                except Exception as e:
                    st.error(f"API 调用失败: {e}")
                    st.stop()
                    
        if loop_count >= MAX_LOOPS and not conclusion_reached:
            st.warning("⚠️ 团队讨论达到最高安全轮次 (10轮) 限制，为防止 API 消耗过大强行终止。")

    st.rerun()