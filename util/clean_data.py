import re

def clean_title(title):
    clean_title=re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]','',title.strip())
    if len(clean_title) > 50:
        clean_title = clean_title[:50]
    if not clean_title:
        return "untitled"
    return clean_title

def reformat_txt(data):
    """
    将txt整理为：
    【标题】
    【时间】
    【来源】
    【正文内容】
    :param data:
    :return:
    """
    title=data.get('title', '无标题')
    pub_time=data.get('pub_time', '无时间')
    content=data.get('content', '无内容')
    source=data.get('source', '无来源')
    new_content=f"[标题]:{title}\n[时间]:{pub_time}\n[来源]:{source}\n\n{content}"
    return new_content

def change_sparse_int(sparse_embeddings):
    new_sparse_embeddings=[]
    for sparse_embedding in sparse_embeddings:
        new_sparse_embedding = {}
        for key,value in sparse_embedding.items():
            try:
                new_key=int(key)
                new_sparse_embedding[new_key]=value
            except ValueError:
                print(f"[ERROR] sparse_embedding key {key} couldn't be converted to integer.")
                return None
        new_sparse_embeddings.append(new_sparse_embedding)
    return new_sparse_embeddings

if __name__=="__main__":
    text="这个水电站项目，正“强渡大渡河”"
    print(clean_title(text))