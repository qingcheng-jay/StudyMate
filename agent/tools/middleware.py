"""
Agent 中间件：工具执行监控、模型调用日志、工具轮次上限（防死循环）
"""
from langchain.agents.middleware import after_model, before_model, wrap_tool_call
from langchain.agents.middleware import AgentState
from langchain.agents.middleware import ModelRequest
from langchain_core.messages import AIMessage, ToolMessage
from langchain.tools.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command
from typing import Callable

from utils.config_handler import agent_conf
from utils.logger_handler import logger


@wrap_tool_call
def monitor_tool(
        # 请求的数据封装
        request: ToolCallRequest,
        # 执行的函数本身
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:             # 工具执行的监控
    logger.info(f"[tool monitor]执行工具：{request.tool_call['name']}")
    logger.info(f"[tool monitor]传入参数：{request.tool_call['args']}")

    try:
        result = handler(request)
        logger.info(f"[tool monitor]工具{request.tool_call['name']}调用成功")
        return result
    except Exception as e:
        logger.error(f"工具{request.tool_call['name']}调用失败，原因：{str(e)}")
        raise e


@before_model
def log_before_model(
        state: AgentState,          # 整个Agent智能体中的状态记录
        runtime: Runtime,           # 记录了整个执行过程中的上下文信息
):         # 在模型执行前输出日志
    logger.info(f"[log_before_model]即将调用模型，带有{len(state['messages'])}条消息。")

    return None


@after_model(can_jump_to=["end"])
def limit_tool_rounds(
        state: AgentState,
        runtime: Runtime,
):         # 工具轮次上限：已达 max_rounds 轮仍要求调工具时直接结束（防死循环）
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        rounds = sum(
            1
            for m in state["messages"][:-1]
            if isinstance(m, AIMessage) and getattr(m, "tool_calls", None)
        )
        if rounds >= agent_conf["max_rounds"]:
            logger.warning(f"[limit_tool_rounds]工具轮次已达上限{agent_conf['max_rounds']}轮，强制结束")
            return {"jump_to": "end"}

    return None
