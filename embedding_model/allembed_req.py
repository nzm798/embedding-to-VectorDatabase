import requests
from typing import List, Tuple, Optional

from embedding_model import BaseEmbeddingClient


class AllEmbeddingClient(BaseEmbeddingClient):
    def __init__(self, api_host: str, api_port):
        super().__init__()
        self.api_host = api_host
        self.api_port = api_port

        self.headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json'
        }
    def embed(self, texts: List[str]):
        """
        批量请求文本的 embedding，分别返回 dense_embeddings列表
        :param texts: 文本列表
        :return: dense_embeddings
        """
        payload = {"sentences": texts}
        url = f'http://{self.api_host}:{self.api_port}/v2/embeddings'
        response = requests.request("POST", url,  headers=self.headers,json=payload)
        dense_embeddings = [_dic["dense_embedding"] for _dic in response.json()["data"]]
        return dense_embeddings

    def embed_all(self, texts: List[str]):
        """
        批量请求文本的 embedding，分别返回 dense_embeddings 和 sparse_embeddings 两个列表
        :param texts: 文本列表
        :return: (dense_embeddings, sparse_embeddings)
        """
        payload = {"sentences": texts}
        url = f'http://{self.api_host}:{self.api_port}/v2/embeddings'
        response = requests.request("POST", url,  headers=self.headers,json=payload)
        dense_embeddings = [_dic["dense_embedding"] for _dic in response.json()["data"]]
        sparse_embeddings= [_dic["sparse_embedding"] for _dic in response.json()["data"]]
        return dense_embeddings, sparse_embeddings

if __name__=="__main__":
    text=["我是一个配角、一个小配角！"]
    client=AllEmbeddingClient("192.168.35.240",7200)
    dense,sparse=client.embed_all(text)
    print(dense)