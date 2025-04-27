from typing import List, Optional


class EmbeddingClient:
    def __init__(self, api_host: str, api_port, api_key: Optional[str] = None):
        self.api_host = api_host
        self.api_port = api_port
        
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json"
        }
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
