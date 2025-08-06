from openai import OpenAI
from abc import ABC, abstractmethod
from typing import Optional

class BaseAPI(ABC):
    def __init__(self, base_url: str, model_name: str,api_key: Optional[str] = ""):
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.client = OpenAI(base_url=base_url, api_key=api_key)
    @abstractmethod
    def chat(self,query:str,**kwargs):
        pass