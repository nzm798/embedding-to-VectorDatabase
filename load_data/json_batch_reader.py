import polars as pl
import threading

from load_data.base_batch_reader import BaseBatchReader
from util.clean_data import clean_title

#def clean_title(title):
#    import re
#    clean_title=re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]','',title.strip())
#    if len(clean_title) > 50:
#        clean_title = clean_title[:50]
#    if not clean_title:
#        return "untitled"
#    return clean_title
class JsonlBatchReader(BaseBatchReader):
    def __init__(self, file_path: str, start: int = 1, end: int = None, batch_size: int = 1000):
        """
        :param file_path: JSONL文件路径
        :param start: 开始读取的行号（包括）从1开始的表示从实际的第几行开始
        :param end: 结束读取的行号（包括），None表示到文件末尾
        :param batch_size: 每批返回多少行
        """
        self.file_path = file_path
        self.start = start-1
        self.end = end
        self.batch_size = batch_size

        self.lock = threading.Lock()  # 用来保证多线程安全取块
        self._load_data()
        self.current_idx = 0
        self.global_idx = start # 初始化为开始读取的行号

    def _load_data(self):
        print(f"[INFO] Loading JSONL file: {self.file_path}")
        df = pl.read_ndjson(self.file_path)
        print(f"[INFO] Total rows: {df.height}")

        # 限制读取范围
        if self.end is None or self.end > df.height:
            self.end = df.height
        self.df = df.slice(self.start, self.end)
        print(f"[INFO] Loaded rows {self.start+1} to {self.end}")

    def has_next_batch(self) -> bool:
        """
        是否还有未读完的批次
        """
        with self.lock:
            return self.global_idx <= self.df.height

    def next_batch(self) -> pl.DataFrame:
        """
        获取下一批数据（线程安全）
        """
        with self.lock:
            # 计算剩余的行数
            remaining_rows = self.df.height - self.global_idx + 1

            # 如果剩余的行数小于batch_size，只读取剩余的数据
            if remaining_rows <= 0:
                return None  # 如果没有数据了，返回 None

            batch_size = min(self.batch_size, remaining_rows)
            # 获取当前批次数据
            batch = self.df.slice(self.current_idx, batch_size)
            self.current_idx += batch_size
            titles=batch["title"]
            # 将每一行的文件名设置为 title_num.txt，其中 num 是该行的全局行号
            file_names = [
                f"{clean_title(titles[i])}_{self.global_idx + i}.pdf" for i in range(batch_size)
            ]

            # 将文件名作为新的一列加入到批次数据中
            batch = batch.with_columns(pl.Series("file_name", file_names))

            # 更新全局行号
            self.global_idx += batch_size

            return batch

if __name__ == "__main__":
    # 初始化 JsonlBatchReader
    reader = JsonlBatchReader(file_path="../news_corpus20250321.jsonl", start=2079262,batch_size=1000)
    import pymysql
    connection = pymysql.connect(
            host="192.168.35.231",
            port=3306,
            user="szzf",
            password="jRzZHvnjRm1kJ9fRj5SL",
            database="dimension_beijing_xicheng",
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=True  # 自动提交，避免忘了commit
        )
    
    # 读取批次数据并打印输出
    batch_idx = 1
    while reader.has_next_batch():
        batch = reader.next_batch()
        print(f"Batch {batch_idx}:")
        # print(batch)
        file_name=batch['file_name'][0]
        print(f"SELECT id FROM `knowledge_document_library` WHERE name = {file_name} LIMIT 1")
        with connection.cursor() as cursor:
            sql = f"SELECT id FROM `knowledge_document_library` WHERE name = %s LIMIT 1"
            cursor.execute(sql, (file_name,))
            print(f"SELECT id FROM `knowledge_document_library` WHERE name = {file_name} LIMIT 1")
            result = cursor.fetchone()
            if result:
                print(result['id']) 
            else:
                print(None)

        batch_idx += 1
