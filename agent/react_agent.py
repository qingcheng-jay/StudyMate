"""
StudyMate ReactAgent：langchain create_agent 组织「LLM 决策 ⇄ 工具执行」的循环。

- system_prompt 从 prompts/main_prompt.txt 加载（utils/prompt_loader）
- 7 个工具由 @tool 声明，中间件负责日志与工具轮次上限（防死循环）
- run_agent(history) 对外接口与原版一致：{"reply": 最终回答, "trace": 本次调用的工具列表}
"""
from langchain.agents import create_agent
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from agent.tools.agent_tools import TOOLS
from agent.tools.middleware import limit_tool_rounds, log_before_model, monitor_tool
from model.factory import get_chat_model, map_llm_exception
from utils.logger_handler import logger
from utils.prompt_loader import load_system_prompts


class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model=get_chat_model(),
            system_prompt=load_system_prompts(),
            tools=TOOLS,
            middleware=[monitor_tool, log_before_model, limit_tool_rounds],
        )

    def execute(self, messages: list[dict]) -> dict:
        """输入前端 dict 消息历史，返回 {"reply", "trace"}。"""
        result = self.agent.invoke({"messages": _to_messages(messages)})

        return {
            "reply": _extract_reply(result["messages"]),
            "trace": _extract_trace(result["messages"]),
        }


def _to_messages(raw: list[dict]) -> list[AnyMessage]:
    """前端 dict 消息 → LangChain 消息对象（System Prompt 由 create_agent 注入）。"""
    out: list[AnyMessage] = []
    for m in raw:
        role, content = m.get("role"), m.get("content", "")
        if role == "user":
            out.append(HumanMessage(content=content))
        elif role == "assistant":
            out.append(AIMessage(content=content))
        elif role == "system":
            out.append(SystemMessage(content=content))
    return out


def _extract_reply(messages: list) -> str:
    """最终回答：末条消息的文本内容；轮次用尽仍要求调工具时给出固定提示。"""
    final = messages[-1]
    if getattr(final, "tool_calls", None):
        # 走到这里说明轮次用尽仍要求调工具
        return "工具调用轮次过多，已停止。请把需求说得更具体一些。"

    content = final.content
    return content if isinstance(content, str) and content else "(模型未返回内容)"


def _extract_trace(messages: list) -> list[dict]:
    """从消息流提取工具调用轨迹：[{tool, args, result}]（前端展示用）。"""
    by_id: dict[str, dict] = {}
    for m in messages:
        for tc in getattr(m, "tool_calls", None) or []:
            by_id[tc["id"]] = {"tool": tc["name"], "args": tc["args"], "result": ""}

    for m in messages:
        if isinstance(m, ToolMessage):
            entry = by_id.get(m.tool_call_id)
            if entry is not None and not entry["result"]:
                content = m.content
                entry["result"] = (content if isinstance(content, str) else str(content))[:400]

    return list(by_id.values())


_agent = None


def _get_agent() -> ReactAgent:
    """惰性构造 Agent（只构造一次；配置缺失时不缓存，下次调用重试）。"""
    global _agent
    if _agent is None:
        _agent = ReactAgent()
    return _agent


def run_agent(messages: list[dict]) -> dict:
    """运行 Agent：返回 {"reply": 最终回答, "trace": 本次调用的工具列表}。"""
    try:
        return _get_agent().execute(messages)
    except Exception as e:  # 把 openai 异常翻译成 LLMCallError；其他异常原样抛出
        mapped = map_llm_exception(e)
        if mapped is e:
            raise
        raise mapped from e


if __name__ == '__main__':
    for chunk in run_agent([{"role": "user", "content": "你好，简单介绍下你自己"}]).values():
        print(chunk if isinstance(chunk, str) else chunk)
