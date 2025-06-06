import requests
from typing import List, Tuple, Optional


class TeiEmbeddingClient:
    def __init__(self, api_host: str, api_port, api_key: Optional[str] = None):
        self.api_host = api_host
        self.api_port = api_port

        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json"
        }
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def embed_all(self, texts: List[str]) -> Tuple[List[List[float]], List[List[float]]]:
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

        dense_embeddings = result
        # 测试，之后获取正式的方法后修改
        sparse_embeddings = [
            {
                364: 0.17531773447990417,
                418: 0.145879546621,
                630: 0.1101302548795,
                3172: 0.268978546412,
                5357: 0.254789645874,
                15483: 0.215479896454225454
            } for _ in range(len(texts))
        ]



        return dense_embeddings, sparse_embeddings


if __name__ == '__main__':
    tei = TeiEmbeddingClient("127.0.0.1", 8080)
    dense_embedding, sparse_embedding = tei.embed_all(["你好我有一个帽衫", "我要在网上问问", "像个大耳朵矮人"])
