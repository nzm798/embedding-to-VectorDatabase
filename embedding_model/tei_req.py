import requests
from typing import List
from embedding_model.embedding_req import EmbeddingClient


class TeiEmbeddingClient(EmbeddingClient):

    def embed(self, texts: List[str]) -> Tuple[List[List[float]], List[List[float]]]:
        """
        批量请求文本的 embedding，分别返回 dense_embeddings 和 sparse_embeddings 两个列表
        :param texts: 文本列表
        :return: (dense_embeddings, sparse_embeddings)
        """
        payload = {"inputs": texts}
        url = f"http://{self.api_host}:{self.api_port}/embed"
        response = requests.post(url, json=payload, headers=self.headers)

        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

        result = response.json()

        dense_embeddings = []
        sparse_embeddings = []

        for item in result:
            dense_embeddings.append(item["dense_embedding"])
            sparse_embeddings.append(item["sparse_embedding"])

        return dense_embeddings, sparse_embeddings