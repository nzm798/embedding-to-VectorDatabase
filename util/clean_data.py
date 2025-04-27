import re

def clean_title(title):
    clean_title=re.sub(r'^\u4e00-\u9fa5a-zA-Z0-9', '', title)
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