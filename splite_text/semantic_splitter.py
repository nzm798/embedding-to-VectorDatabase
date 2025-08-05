import requests
import asyncio
import aiohttp
import time

from langchain_core.embeddings import Embeddings
from typing import Optional, List, Iterator

import sys
sys.path.append('/workspace')

from splite_text import BaseSplitter
from langchain_experimental.text_splitter import SemanticChunker

class SemanticSplitter(BaseSplitter):
    def __init__(self, embeddings: Embeddings, buffer_size: int = 1, breakpoint_threshold_type="percentile",
                 breakpoint_threshold_amount: Optional[float] = None,
                 number_of_chunks: Optional[int] = None,
                 sentence_split_regex: str = r"(?<=[。.!?！？； ])\s*",
                 min_chunk_size: Optional[int] = None, ):
        """
        按照语义切分器
        Args:
            embeddings (Embeddings):
                用于计算语义相似度的嵌入模型，需要实现embed_documents和embed_query方法
            breakpoint_threshold_type (str, optional):
                切分点阈值类型，默认为"percentile"
                - "percentile": 百分位数阈值
                - "standard_deviation": 标准差阈值
                - "interquartile": 四分位距阈值
                - "gradient": 梯度变化阈值
            breakpoint_threshold_amount (float, optional):
                阈值具体数值，与阈值类型配合使用
                - percentile: 0-100之间的数值，如80表示80%分位数
                - standard_deviation: 标准差倍数，如1.5
                - interquartile: 四分位距倍数，如1.5
            number_of_chunks (int, optional):
                期望切分成的块数，如果设置则忽略阈值设置
            sentence_split_regex (str, optional):
                句子分割的正则表达式，支持中英文标点符号
                默认支持：中文句号。英文句号. 问号? 感叹号! 中文问号？中文感叹号！分号；空格
            min_chunk_size (int, optional):
                每个块的最小字符数，避免产生过小的文本块
        """
        super().__init__()
        self.text_splitter = SemanticChunker(embeddings, buffer_size=buffer_size,
                                             breakpoint_threshold_type=breakpoint_threshold_type,
                                             breakpoint_threshold_amount=breakpoint_threshold_amount,
                                             number_of_chunks=number_of_chunks,
                                             sentence_split_regex=sentence_split_regex, min_chunk_size=min_chunk_size)

    def split(self, text):
        return self.text_splitter.split_text(text)


class TeiEmbeddings(Embeddings):
    def __init__(self,
                 api_host: str,
                 api_port: int,
                 api_key: Optional[str] = None,
                 batch_size: int = 32,
                 timeout: int = 30,
                 max_retries: int = 3,
                 use_async: bool = False):
        """
        初始化TEI嵌入客户端

        Args:
            api_host (str): API主机地址
            api_port (int): API端口
            api_key (Optional[str]): API密钥
            batch_size (int): 批处理大小，默认32
            timeout (int): 请求超时时间（秒），默认30
            max_retries (int): 最大重试次数，默认3
            use_async (bool): 是否使用异步处理，默认False
        """
        self.url = f"http://{api_host}:{api_port}/embed"
        self.batch_size = batch_size
        self.timeout = timeout
        self.max_retries = max_retries
        self.use_async = use_async

        self.headers = {
            "Content-Type": "application/json"
        }
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def _batch_texts(self, texts: List[str]) -> Iterator[List[str]]:
        """
        将文本列表分批
        Args:
            texts (List[str]): 输入文本列表
        Yields:
            List[str]: 批次文本
        """
        for i in range(0, len(texts), self.batch_size):
            yield texts[i:i + self.batch_size]

    def _make_request(self, payload: dict) -> dict:
        """
        发送HTTP请求，包含重试机制
        Args:
            payload (dict): 请求载荷
        Returns:
            dict: 响应数据
        Raises:
            Exception: 请求失败时抛出异常
        """
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    self.url,
                    json=payload,
                    headers=self.headers,
                    timeout=self.timeout
                )

                if response.status_code == 200:
                    return response.json()
                else:
                    print(f"[WARNING] Request failed (attempt {attempt + 1}/{self.max_retries + 1}): "
                          f"Status {response.status_code}, {response.text}")

            except requests.exceptions.RequestException as e:
                print(f"[ERROR] Request exception (attempt {attempt + 1}/{self.max_retries + 1}): {e}")

            # 如果不是最后一次尝试，等待后重试
            if attempt < self.max_retries:
                time.sleep(2 ** attempt)  # 指数退避

        raise Exception(f"Request failed after {self.max_retries + 1} attempts")

    async def _make_async_request(self, session: aiohttp.ClientSession, payload: dict) -> dict:
        """
        异步发送HTTP请求，包含重试机制
        Args:
            session (aiohttp.ClientSession): aiohttp会话
            payload (dict): 请求载荷
        Returns:
            dict: 响应数据
        Raises:
            Exception: 请求失败时抛出异常
        """
        for attempt in range(self.max_retries + 1):
            try:
                async with session.post(
                        self.url,
                        json=payload,
                        headers=self.headers,
                        timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        print(f"[WARNING] Async request failed (attempt {attempt + 1}/{self.max_retries + 1}): "
                              f"Status {response.status}, {await response.text()}")

            except Exception as e:
                print(f"[ERROR] Async request exception (attempt {attempt + 1}/{self.max_retries + 1}): {e}")

            # 如果不是最后一次尝试，等待后重试
            if attempt < self.max_retries:
                await asyncio.sleep(2 ** attempt)  # 指数退避

        raise Exception(f"Async request failed after {self.max_retries + 1} attempts")

    async def _embed_documents_async(self, texts: List[str]) -> List[List[float]]:
        """
        异步批量嵌入文档
        Args:
            texts (List[str]): 输入文本列表
        Returns:
            List[List[float]]: 嵌入向量列表
        """
        if not texts:
            return []

        # 创建所有批次
        batches = list(self._batch_texts(texts))
        all_embeddings = [None] * len(batches)  # 预分配结果列表

        # 创建aiohttp会话
        async with aiohttp.ClientSession() as session:
            # 创建所有任务
            tasks = []
            for batch_idx, batch in enumerate(batches):
                payload = {"inputs": batch}
                task = self._make_async_request(session, payload)
                tasks.append((batch_idx, task))

            # 并发执行所有任务
            for batch_idx, task in tasks:
                try:
                    batch_embeddings = await task
                    if not isinstance(batch_embeddings, list):
                        raise Exception(f"Expected list response, got {type(batch_embeddings)}")
                    all_embeddings[batch_idx] = batch_embeddings
                except Exception as e:
                    print(f"[ERROR] Failed to process batch {batch_idx + 1}: {e}")
                    raise Exception(f"Failed to embed batch {batch_idx + 1}: {e}")
        result = []
        for batch_embeddings in all_embeddings:
            if batch_embeddings:
                result.extend(batch_embeddings)
        return result

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        批量嵌入文档，自动分批处理大量文本
        Args:
            texts (List[str]): 输入文本列表
        Returns:
            List[List[float]]: 嵌入向量列表
        Raises:
            Exception: API请求失败时抛出异常
        """
        if not texts:
            return []

        if self.use_async:
            # 使用异步处理
            return asyncio.run(self._embed_documents_async(texts))
        else:
            # 使用同步处理
            all_embeddings = []
            for batch_idx, batch in enumerate(self._batch_texts(texts)):
                try:
                    payload = {"inputs": batch}
                    batch_embeddings = self._make_request(payload)
                    if not isinstance(batch_embeddings, list):
                        raise Exception(f"Expected list response, got {type(batch_embeddings)}")

                    all_embeddings.extend(batch_embeddings)

                except Exception as e:
                    print(f"[ERROR] Failed to process batch {batch_idx + 1}: {e}")
                    raise Exception(f"Failed to embed batch {batch_idx + 1}: {e}")
            return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        嵌入单个查询文本
        Args:
            text (str): 输入查询文本
        Returns:
            List[float]: 嵌入向量
        Raises:
            Exception: API请求失败时抛出异常
        """
        if not text:
            return []
        try:
            payload = {"inputs": [text]}
            embeddings = self._make_request(payload)
            if not embeddings or not isinstance(embeddings, list):
                raise Exception(f"Invalid response format: {embeddings}")
            return embeddings[0]
        except Exception as e:
            print(f"[ERROR] Failed to embed query: {e}")
            raise Exception(f"Failed to embed query: {e}")


if __name__ == "__main__":
    embeddings = TeiEmbeddings("192.168.100.7", "8082")
    # print(embeddings.embed_query("你好我有一个毛衫"))
    # print(embeddings.embed_documents(["你好我有一个帽衫","我要在网上问问"]))
    splitter = SemanticSplitter(embeddings)
    with open('/workspace/splite_text/test_doc.txt', 'r', encoding="utf-8") as f:
        text = f.read()
    split_text = splitter.split(text)
    for text in split_text:
        print(f"切分长度为{len(text)},切分：{text}\n\n")
