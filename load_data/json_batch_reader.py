import polars as pl
import threading


class JsonlBatchReader:
    def __init__(self, file_path: str, start: int = 0, end: int = None, batch_size: int = 1000):
        """
        :param file_path: JSONL文件路径
        :param start: 开始读取的行号（包括）
        :param end: 结束读取的行号（不包括），None表示到文件末尾
        :param batch_size: 每批返回多少行
        """
        self.file_path = file_path
        self.start = start
        self.end = end
        self.batch_size = batch_size

        self.lock = threading.Lock()  # 用来保证多线程安全取块
        self._load_data()
        self.current_idx = 0

    def _load_data(self):
        print(f"[INFO] Loading JSONL file: {self.file_path}")
        df = pl.read_ndjson(self.file_path)
        print(f"[INFO] Total rows: {df.height}")

        # 限制读取范围
        if self.end is None or self.end > df.height:
            self.end = df.height
        self.df = df.slice(self.start, self.end - self.start)
        print(f"[INFO] Loaded rows {self.start} to {self.end}")

    def has_next_batch(self) -> bool:
        """
        是否还有未读完的批次
        """
        with self.lock:
            return self.current_idx < self.df.height

    def next_batch(self) -> pl.DataFrame:
        """
        获取下一批数据（线程安全）
        """
        with self.lock:
            if self.current_idx >= self.df.height:
                return None  # 没有更多数据了

            batch = self.df.slice(self.current_idx, self.batch_size)
            self.current_idx += self.batch_size
            return batch

