"""
模型工厂：统一构造 LLM 与 Embedding 模型
- LLM 走 .env 的 OpenAI 兼容配置（DeepSeek / 通义千问 / Moonshot / MiMo 等均可）
- Embedding 走通义千问 DashScope 兼容模式（text-embedding-v1，1536 维，与既有向量库兼容）
配置缺失时延迟抛错（get_chat_model / get_embed_model 调用时才校验），保证无 Key 时前端仍可启动。
"""
import os
from abc import ABC, abstractmethod
from functools import lru_cache  #给函数配一个“备忘录”，算过的直接查表，不再重算。

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)

from utils.config_handler import agent_conf, rag_conf

# 两个自定义异常类
class LLMConfigError(Exception):
    """.env 配置缺失或仍是占位值。"""


class LLMCallError(Exception):
    """调用模型服务失败（网络、认证、服务端错误）。"""


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self):
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> ChatOpenAI:
        api_key = os.getenv("LLM_API_KEY", "").strip()
        base_url = os.getenv("LLM_BASE_URL", "").strip()
        model = os.getenv("LLM_MODEL", "").strip()
        if not api_key or api_key == "sk-your-api-key":
            raise LLMConfigError("未配置 LLM_API_KEY，请在 .env 中填写")
        if not base_url:
            raise LLMConfigError("未配置 LLM_BASE_URL，请在 .env 中填写")
        if not model:
            raise LLMConfigError("未配置 LLM_MODEL，请在 .env 中填写")

        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=agent_conf["temperature"],
            timeout=agent_conf["timeout"],
        )


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> OpenAIEmbeddings:
        api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
        if not api_key or api_key == "sk-your-dashscope-key":
            raise RuntimeError("未配置 DASHSCOPE_API_KEY，请在 .env 中填写")

        return OpenAIEmbeddings(
            model=rag_conf["embedding_model_name"],
            base_url=rag_conf["embedding_base_url"],
            api_key=api_key,
            chunk_size=rag_conf["embedding_batch_size"],
            # DashScope 兼容模式要求接收原文，不能发本地 tokenize 后的 token ids     接口协议兼容但默认参数不兼容
            check_embedding_ctx_length=False,
        )


@lru_cache(maxsize=1)  # 缓存函数结果，同样的参数不再重算
def get_chat_model() -> ChatOpenAI:
    """LLM 单例（配置缺失时抛 LLMConfigError）。"""
    return ChatModelFactory().generator()


@lru_cache(maxsize=1)
def get_embed_model() -> OpenAIEmbeddings:
    """Embedding 单例（配置缺失时抛 RuntimeError）。"""
    return EmbeddingsFactory().generator()


def map_llm_exception(e: Exception) -> Exception:
    """把 openai 库的异常翻译成对用户友好的 LLMCallError。"""
    # isinstance(对象, 类型)
    if isinstance(e, AuthenticationError):
        return LLMCallError("API Key 无效或已过期，请检查 .env 配置")
    if isinstance(e, RateLimitError):  # 注意：RateLimitError 是 APIError 子类，必须先于 APIError 判断
        return LLMCallError("请求频率超限或账户额度不足（429），请稍后再试或检查服务商额度")
    if isinstance(e, (APIConnectionError, APITimeoutError)):
        return LLMCallError("无法连接模型服务，请检查 LLM_BASE_URL 与网络")
    if isinstance(e, APIError):
        return LLMCallError(f"模型服务返回错误：{e}")
    return e


if __name__ == '__main__':
    model = get_chat_model()
    print(model.invoke("用一句话介绍你自己").content[:100])
