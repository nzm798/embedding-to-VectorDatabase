import requests
import time
import aiohttp
import asyncio
from requests import session
from typing import List, Optional, Iterable
from embedding_model.base_req import BaseEmbeddingClient


class TeiEmbeddingClient(BaseEmbeddingClient):
    def __init__(self, dense_api_host: str, dense_api_port, sparse_api_host: Optional[str] = None, sparse_api_port=None,
                 api_key: Optional[str] = None, batch_size: int = 64, timeout: int = 30, max_retries: int = 3,
                 use_async: bool = True):
        super().__init__()
        self.dense_api_host = dense_api_host
        self.dense_api_port = dense_api_port

        if sparse_api_host and sparse_api_port:
            self.sparse_api_host = sparse_api_host
            self.sparse_api_port = sparse_api_port
        else:
            self.sparse_api_host = dense_api_host
            self.sparse_api_port = dense_api_port

        self.batch_size = batch_size
        self.timeout = timeout
        self.max_retries = max_retries
        self.use_async = use_async
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json"
        }
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def _batch_texts(self, texts: List[str]) -> Iterable[List[str]]:
        # Batch the text list
        for i in range(0, len(texts), self.batch_size):
            yield texts[i:i + self.batch_size]

    def _make_request(self, url: str,payload: dict) -> dict:

        # Send HTTP requests with a retry mechanism

        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(url, json=payload, headers=self.headers, timeout=self.timeout)
                if response.status_code == 200:
                    return response.json()
                else:
                    print(f"[WARNING] Request failed (attempt {attempt + 1}/{self.max_retries + 1}): "
                          f"Status {response.status_code}, {response.text}")
            except requests.exceptions.RequestException as e:
                print(f"[ERROR] Request exception (attempt {attempt + 1}/{self.max_retries + 1}): {e}")
            if attempt < self.max_retries:
                time.sleep(2 ** attempt)

    async def _amake_request(self, session: aiohttp.ClientSession, url: str,payload: dict) -> dict:
        # Asynchronously process requests for retries
        for attempt in range(self.max_retries + 1):
            try:
                async with session.post(
                    url,
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
            if attempt < self.max_retries:
                await asyncio.sleep(2 ** attempt)
        raise Exception(f"Async request failed after {self.max_retries + 1} attempts")

    async def _aembed(self,texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        batches = list(self._batch_texts(texts))
        all_dense_embeddings = [None] * len(batches)
        dense_url = f"http://{self.dense_api_host}:{self.dense_api_port}/embed"

        async with aiohttp.ClientSession() as session:
            tasks = []
            for batch_idx , batch in enumerate(batches):
                payload = {"inputs": batch}
                task = self._amake_request(session, dense_url, payload)
                tasks.append((batch_idx, task))
            for batch_idx, task in tasks:
                try:
                    batch_embeddings = await task
                    if not isinstance(batch_embeddings, list):
                        raise Exception(f"Expected list response, got {type(batch_embeddings)}")
                    all_dense_embeddings[batch_idx] = batch_embeddings
                except Exception as e:
                    print(f"[ERROR] Failed to process batch {batch_idx + 1}: {e}")
                    raise Exception(f"Failed to embed batch {batch_idx + 1}: {e}")
        result = []
        for batch_embeddings in all_dense_embeddings:
            if batch_embeddings:
                result.extend(batch_embeddings)
        return result

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        批量请求文本的 embedding，分别返回 dense_embeddings 列表
        :param texts: 文本列表
        :return: dense_embeddings
        """
        if not texts:
            return []
        if self.use_async:
            return asyncio.run(self._aembed(texts))
        else:
            dense_url = f"http://{self.dense_api_host}:{self.dense_api_port}/embed"
            all_dense_embeddings = []
            for batch_idx, batch in enumerate(self._batch_texts(texts)):
                try:
                    payload = {"inputs": batch}
                    batch_embeddings = self._make_request(dense_url,payload)
                    if not isinstance(batch_embeddings, list):
                        raise Exception(f"Expected list response, got {type(batch_embeddings)}")
                    all_dense_embeddings.extend(batch_embeddings)
                except Exception as e:
                    print(f"[ERROR] Failed to process batch {batch_idx + 1}: {e}")
                    raise Exception(f"Failed to embed batch {batch_idx + 1}: {e}")

        return all_dense_embeddings

    async def _aembed_all(self,texts: List[str]):
        if not texts:
            return [],[]
        batches = list(self._batch_texts(texts))
        all_dense_embeddings = [None] * len(batches)
        all_sparse_embeddings = [None] * len(batches)

        dense_url = f"http://{self.dense_api_host}:{self.dense_api_port}/embed"
        sparse_url = f"http://{self.sparse_api_host}:{self.sparse_api_port}/embed_sparse"

        async with aiohttp.ClientSession() as session:
            all_tasks = []
            for batch_idx, batch in enumerate(batches):
                payload = {"inputs": batch}
                task_denses = self._amake_request(session, dense_url, payload)
                task_sparse = self._amake_request(session, sparse_url, payload)
                all_tasks.append((task_denses, task_sparse))
            for batch_idx, (dense_task,sparse_task) in enumerate(all_tasks):
                try:
                    dense_embeddings = await dense_task
                    sparse_embeddings = await sparse_task
                    if not isinstance(dense_embeddings, list) or not isinstance(sparse_embeddings, list):
                        raise Exception(f"Expected list response, got ({type(dense_embeddings)},{type(sparse_embeddings)})")
                    all_dense_embeddings.extend(dense_embeddings)
                    all_sparse_embeddings.extend(sparse_embeddings)
                except Exception as e:
                    print(f"[ERROR] Failed to process batch {batch_idx + 1}: {e}")
                    raise Exception(f"Failed to embed batch {batch_idx + 1}: {e}")
        dense_result = []
        sparse_result = []
        for dense_embeddings, sparse_embeddings in zip(all_dense_embeddings, all_sparse_embeddings):
            if dense_embeddings and sparse_embeddings:
                dense_result.append(dense_embeddings)
                sparse_result.append(sparse_embeddings)
        return dense_result, sparse_result


    def embed_all(self, texts: List[str]):
        """
        批量请求文本的 embedding，分别返回 dense_embeddings 和 sparse_embeddings 两个列表
        :param texts: 文本列表
        :return: (dense_embeddings, sparse_embeddings)
        """
        if not texts:
            return [], []
        if self.use_async:
            return asyncio.run(self._aembed_all(texts))
        else:
            dense_url = f"http://{self.dense_api_host}:{self.dense_api_port}/embed"
            sparse_url = f"http://{self.sparse_api_host}:{self.sparse_api_port}/embed_sparse"
            all_dense_embeddings = []
            all_sparse_embeddings = []
            for batch_idx, batch in enumerate(self._batch_texts(texts)):
                try:
                    payload = {"inputs": batch}
                    dense_embeddings = self._make_request(dense_url, payload)
                    sparse_embeddings = self._make_request(sparse_url, payload)
                    if not isinstance(dense_embeddings, list) or not isinstance(sparse_embeddings, list):
                        raise Exception(f"Expected list response, got ({type(dense_embeddings)},{type(sparse_embeddings)})")
                    all_dense_embeddings.extend(dense_embeddings)
                    all_sparse_embeddings.extend(sparse_embeddings)
                except Exception as e:
                    print(f"[ERROR] Failed to process batch {batch_idx + 1}: {e}")
                    raise Exception(f"Failed to embed batch {batch_idx + 1}: {e}")
            return all_dense_embeddings, all_sparse_embeddings


if __name__ == '__main__':
    tei = TeiEmbeddingClient("192.168.100.7", 8082, sparse_api_host="192.168.100.7", sparse_api_port=8181,use_async=True)
    # dense_embeddings, sparse_embedding = tei.embed_all(["你好我有一个帽衫", "我要在网上问问", "像个大耳朵矮人"])
    dense_embeddings = tei.embed(["你好我有一个帽衫", "我要在网上问问", "像个大耳朵矮人"])
    print(dense_embeddings)
