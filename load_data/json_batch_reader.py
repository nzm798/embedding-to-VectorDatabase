import polars as pl
import threading

from util.clean_data import clean_title


class JsonlBatchReader:
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
                f"{clean_title(titles[i])}_{self.global_idx + i}.txt" for i in range(batch_size)
            ]

            # 将文件名作为新的一列加入到批次数据中
            batch = batch.with_columns(pl.Series("file_name", file_names))

            # 更新全局行号
            self.global_idx += batch_size

            return batch

if __name__ == "__main__":
    # 初始化 JsonlBatchReader
    reader = JsonlBatchReader(file_path="test_data.jsonl", start=7,end=26, batch_size=5)

    # 读取批次数据并打印输出
    batch_idx = 1
    while reader.has_next_batch():
        batch = reader.next_batch()
        print(f"Batch {batch_idx}:")
        print(batch)

        batch_idx += 1