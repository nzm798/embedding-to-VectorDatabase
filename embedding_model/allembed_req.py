import requests
from typing import List, Tuple, Optional


class AllEmbeddingClient:
    def __init__(self, api_host: str, api_port):
        self.api_host = api_host
        self.api_port = api_port

        self.headers = {
            "Content-Type": "application/json",
            "accept": "application/json"
        }

    def embed_all(self, texts: List[str]):
        """
        批量请求文本的 embedding，分别返回 dense_embeddings 和 sparse_embeddings 两个列表
        :param texts: 文本列表
        :return: (dense_embeddings, sparse_embeddings)
        """
        payload = {"sentences": texts}
        url = f"http://{self.api_host}:{self.api_port}/v2/embeddings"
        response = requests.post("POST",url, json=payload, headers=self.headers)
        dense_embeddings = [_dic["dense_embedding"] for _dic in response.json()["data"]]
        sparse_embeddings= [_dic["sparse_embedding"] for _dic in response.json()["data"]]
        return dense_embeddings, sparse_embeddings