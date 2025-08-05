import requests
from typing import List, Optional

from embedding_model.base_req import BaseEmbeddingClient


class TeiEmbeddingClient (BaseEmbeddingClient):
    def __init__(self, dense_api_host: str, dense_api_port, sparse_api_host:Optional[str] = None,sparse_api_port = None,api_key: Optional[str] = None):
        super().__init__()
        self.dense_api_host = dense_api_host
        self.dense_api_port = dense_api_port

        if sparse_api_host and sparse_api_port:
            self.sparse_api_host = sparse_api_host
            self.sparse_api_port = sparse_api_port
        else:
            self.sparse_api_host = dense_api_host
            self.sparse_api_port = dense_api_port

        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json"
        }
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def embed(self, texts: List[str]) -> List[float]:
        """
        批量请求文本的 embedding，分别返回 dense_embeddings 列表
        :param texts: 文本列表
        :return: dense_embeddings
        """
        payload = {"inputs": texts}
        dense_url = f"http://{self.dense_api_host}:{self.dense_api_port}/embed"
        dense_response = requests.post(dense_url, json=payload, headers=self.headers)
        if dense_response.status_code != 200:
            raise Exception(f"Request failed: {dense_response.status_code}, {dense_response.text}")

        dense_embeddings = dense_response.json()

        return dense_embeddings

    def embed_all(self, texts: List[str]):
        """
        批量请求文本的 embedding，分别返回 dense_embeddings 和 sparse_embeddings 两个列表
        :param texts: 文本列表
        :return: (dense_embeddings, sparse_embeddings)
        """
        payload = {"inputs": texts}
        dense_url = f"http://{self.dense_api_host}:{self.dense_api_port}/embed"
        dense_response = requests.post(dense_url, json=payload, headers=self.headers)
        if dense_response.status_code != 200:
            raise Exception(f"Request failed: {dense_response.status_code}, {dense_response.text}")

        dense_embeddings = dense_response.json()

        sparse_url = f"http://{self.sparse_api_host}:{self.sparse_api_port}/embed_sparse"
        sparse_response = requests.post(sparse_url, json=payload, headers=self.headers)

        if sparse_response.status_code != 200:
            raise Exception(f"Request failed: {sparse_response.status_code}, {sparse_response.text}")

        sparse_objs = sparse_response.json()
        sparse_embeddings = []
        for sparse_obj in sparse_objs:
            sparse_embedding = {}
            for i in sparse_obj:
                new_key = int(i.get("index"))
                sparse_embedding[new_key] = i.get("value")
            sparse_embeddings.append(sparse_embedding)

        return dense_embeddings, sparse_embeddings


if __name__ == '__main__':
    tei = TeiEmbeddingClient("192.168.100.7", 8082,sparse_api_host="192.168.100.7",sparse_api_port=8181)
    dense_embeddings, sparse_embedding = tei.embed_all(["你好我有一个帽衫", "我要在网上问问", "像个大耳朵矮人"])
    print(dense_embeddings)
    print(sparse_embedding)
