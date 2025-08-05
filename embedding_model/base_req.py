from abc import ABC,abstractmethod
from typing import List

class BaseEmbeddingClient(ABC):
    def __init__(self ,**kwargs):
        pass
    @abstractmethod
    def embed(self, **kwargs) -> List[float]:
        pass
    @abstractmethod
    def embed_all(self, **kwargs):
        pass