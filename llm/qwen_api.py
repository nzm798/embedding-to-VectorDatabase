import copy
import asyncio
from typing import Optional, Dict, List, Any

import sys
sys.path.append('/workspace')
from llm.base_api import BaseAPI
from util.utils import iter_response


class QwenAPI(BaseAPI):
    def __init__(self, base_url: str, model_name: str, api_key: Optional[str] = ""):
        super().__init__(base_url, model_name, api_key)
        self.req_dic = {
            'model': model_name,
            'max_tokens': 1000,
            'stream': False,
            "temperature": 0,
            "frequency_penalty": 0.1,
            "top_p": 0.1,
        }

        self.system = """你是一个数据处理打标专家。"""

    async def _achat(self,query: str, system=None, history: Optional[List] = [], stream=False, req_params: Optional[Dict] = {},
             extra_body: Optional[Dict] = None):
        req_dic = copy.deepcopy(self.req_dic)
        if req_params:
            req_dic.update(req_params)
        msgs = []
        if system and len(system) > 0:
            sys_dic = {"role": "system", "content": system}
            msgs.append(sys_dic)
        elif self.system:
            sys_dic = {"role": "system", "content": self.system}
            msgs.append(sys_dic)
        for tup in history:
            q, a = tup[:2]
            if q != query:
                msgs.append({"role": "user", "content": q})
                msgs.append({"role": "assistant", "content": a})
        msgs.append({"role": "user", "content": query})
        req_dic["messages"] = msgs
        req_dic["stream"] = stream
        response = self.client.chat.completions.create(**req_dic, extra_body=extra_body)
        if stream:
            return iter_response(response)
        else:
            return response.choices[0].message.content


    def chat(self, query: str, system=None, history: Optional[List] = [], stream=False, req_params: Optional[Dict] = {},
             extra_body: Optional[Dict] = None):
        return asyncio.run(self._achat(query, system, history, stream, req_params, extra_body))

if __name__ == '__main__':
    qwen = QwenAPI(base_url='http://192.168.100.7:1025/v1/', model_name='qwen3-14b')
    print(qwen.chat("你好,很高兴认识你。/no_think",stream=True),flush=True)