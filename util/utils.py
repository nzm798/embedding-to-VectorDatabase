def iter_response(response):
    token_num=0
    res = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            res += chunk.choices[0].delta.content
            token_num+=1
            yield res