import requests
from typing import List


class MyEmbeddingClient:
    def __init__(self, api_host: str, api_port):
        self.api_host = api_host
        self.api_port = api_port

        self.headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json'
        }

    def embed_all(self, texts: List[str]):
        """
        批量请求文本的 embedding，分别返回 dense_embeddings 和 sparse_embeddings 两个列表
        :param texts: 文本列表
        :return: (dense_embeddings, sparse_embeddings)
        """
        payload = {"sentences": texts}
        url = f'http://{self.api_host}:{self.api_port}/embeddings'
        response = requests.request("POST", url,  headers=self.headers,json=payload)
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}--{response.text}")
        dense_embeddings = response.json()['embeddings']
        sparse_embeddings= response.json()['sparse_embeddings']
        return dense_embeddings, sparse_embeddings

if __name__=="__main__":
    text=["我是一个配角、一个小配角！"]
    client=MyEmbeddingClient("106.63.5.25",7300)
    dense,sparse=client.embed_all(text)
    print(sparse)