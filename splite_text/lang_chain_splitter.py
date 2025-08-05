from langchain.text_splitter import RecursiveCharacterTextSplitter

from splite_text.base_splitter import BaseSplitter


class TextSplitter(BaseSplitter):
    def __init__(self, chunk_size: int = 1024, overlap: int = 100):
        """
        初始化文本分块器
        :param chunk_size: 每个分块的最大长度
        :param overlap: 分块之间的重叠长度
        """
        super().__init__()
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=["<row>", "</row>", "<Cell>", "</Cell>", "\n", "。", "，", "；", "！", "？", " "],
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap,
            length_function=len,
            keep_separator=False
        )

    def split(self, text: str):
        """
        使用 `RecursiveCharacterTextSplitter` 对文本进行分块
        :param text: 待分块的文本
        :return: 分块后的文本列表
        """
        return self.text_splitter.split_text(text)
