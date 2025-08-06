import threading
from collections import deque
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
from hdfs import InsecureClient
import time
import sys
sys.path.append("/workspace")
from load_data.base_batch_reader import BaseBatchReader


class HdfsBatchReader(BaseBatchReader):
    def __init__(self, hdfs_url: str,
                 hdfs_user: str,
                 batch_size: int = 64,
                 max_concurrent_reads: int = 20,
                 cache_preload_size: int = 128,
                 max_cache_size: int = 256):
        self.hdfs_url = hdfs_url
        self.batch_size = batch_size
        self.max_concurrent_reads = max_concurrent_reads
        self.cache_preload_size = cache_preload_size
        self.max_cache_size = max_cache_size

        self.file_list = []
        self.total_files = 0
        self.current_idx = 0
        self.lock = threading.Lock()

        # 初始化HDFS客户端
        self.hdfs_client = InsecureClient(hdfs_url, user=hdfs_user)

        # 缓存相关
        self.data_cache = {}  # {id: data}
        self.cache_lock = threading.RLock()
        self.cache_condition = threading.Condition(self.cache_lock)

        self.stop_cache_thread = threading.Event()
        self.cache_thread = None

    def set_file_list(self, file_list: List[Dict]):
        with self.lock:
            self.file_list = sorted(file_list, key=lambda x: x.get('id', 0))
            self.total_files = len(self.file_list)
            self.current_idx = 0

    def start_cache_preload(self, file_list: List[Dict]):
        if not file_list:
            print("[ERROR] file_list is empty")
            return
        self.set_file_list(file_list)
        if self.cache_thread is None or not self.cache_thread.is_alive():
            self.stop_cache_thread.clear()
            self.cache_thread = threading.Thread(target=self._preload_cache_worker, daemon=True)
            self.cache_thread.start()
            print(f"[INFO] Cache preload thread started, max cache size: {self.max_cache_size}")

    def _preload_cache_worker(self):
        print("[INFO] Starting cache preload worker")
        preload_idx = 0

        while not self.stop_cache_thread.is_set() and preload_idx < self.total_files:
            with self.cache_condition:
                # 等待缓存未满
                while len(self.data_cache) >= self.max_cache_size and not self.stop_cache_thread.is_set():
                    print(f"[INFO] Cache full ({len(self.data_cache)}), waiting...")
                    self.cache_condition.wait(timeout=1)

                if self.stop_cache_thread.is_set():
                    break

                end_idx = min(preload_idx + self.cache_preload_size, self.total_files)
                preload_files = self.file_list[preload_idx:end_idx]
                print(f"[INFO] Preloading files {preload_idx + 1} to {end_idx}")

                preload_idx = end_idx

            # 并发加载
            with ThreadPoolExecutor(max_workers=self.max_concurrent_reads) as executor:
                futures = {executor.submit(self._read_single_file_hdfs, f): f for f in preload_files}
                for future in futures:
                    try:
                        result = future.result(timeout=30)
                        if result:
                            with self.cache_condition:
                                self.data_cache[result['id']] = result
                                self.cache_condition.notify_all()  # 唤醒等待线程
                    except Exception as e:
                        file_info = futures[future]
                        print(f"[ERROR] Failed to preload file {file_info.get('id')}: {e}")

        print("[INFO] Cache preload worker finished")

    def _read_single_file_hdfs(self, file_info: Dict) -> Optional[Dict]:
        try:
            file_path_pdf = file_info.get('file_info_url')
            if not file_path_pdf:
                return None
            if file_path_pdf.lower().endswith('.pdf'):
                file_path = file_path_pdf[:-4] + '.txt'
            else:
                print(f"[ERROR] File info url: {file_path_pdf} does not end with '.pdf']")

            with self.hdfs_client.read(file_path) as reader:
                content = reader.read().decode('utf-8')

            return {
                'id': file_info.get('id'),
                'name': file_info.get('name'),
                'file_path': file_path_pdf,
                'content': content,
                'word_count': len(content),
                'file_name': file_info.get('name'),
            }
        except Exception as e:
            print(f"[ERROR] Failed to read HDFS file {file_info.get('file_info_url')}: {e}")
            return None

    def has_next_batch(self) -> bool:
        with self.lock:
            return self.current_idx < self.total_files

    def next_batch(self) -> Optional[List[Dict]]:
        with self.lock:
            if self.current_idx >= self.total_files:
                return None
            end_idx = min(self.current_idx + self.batch_size, self.total_files)
            batch_files = self.file_list[self.current_idx:end_idx]
            self.current_idx = end_idx

        batch_results = []
        with self.cache_condition:
            for file_info in batch_files:
                file_id = file_info.get('id')
                while file_id not in self.data_cache and not self.stop_cache_thread.is_set():
                    print(f"[INFO] Waiting for file {file_id} to load into cache...")
                    self.cache_condition.wait(timeout=1)
                if file_id in self.data_cache:
                    batch_results.append(self.data_cache.pop(file_id))  # 读取并删除

        return batch_results if batch_results else None

    def get_cache_stats(self) -> Dict:
        with self.cache_lock:
            return {
                'cached_files': len(self.data_cache),
                'total_files': self.total_files,
                'cache_hit_rate': len(self.data_cache) / self.total_files if self.total_files > 0 else 0
            }

    def close(self):
        print("[INFO] Closing HdfsBatchReader")
        self.stop_cache_thread.set()
        with self.cache_condition:
            self.cache_condition.notify_all()
        if self.cache_thread and self.cache_thread.is_alive():
            self.cache_thread.join(timeout=5)
        with self.cache_lock:
            self.data_cache.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

if __name__ == '__main__':
    import sys
    sys.path.append('/workspace')
    from vector_database import MySQLClient
    mysql_client = MySQLClient(
        host="192.168.100.9",
        port=3306,
        user="szzf",
        password="jRzZHvnjRm1kJ9fRj5SL",
        database="dimension_beijing_xicheng",
        table_name="knowledge_document_library"
    )
    conn, cursor = mysql_client.get_conn()
    file_list = mysql_client.get_files_by_knowledge(123, conn, cursor)
    mysql_client.close()

    hdfs_reader = HdfsBatchReader(hdfs_url="http://192.168.100.9:9870",hdfs_user="ecm-bf52")
    hdfs_reader.start_cache_preload(file_list)

    batch_idx = 1
    while hdfs_reader.has_next_batch():
        batch = hdfs_reader.next_batch()
        print(f"Batch {batch_idx}:")
        for k in batch:
            print(f"Batch name: {k['name']},Batch id: {k['id']},Batch file path:{k['file_path']},Batch word count : {k['word_count']},Batch size: {len(batch)}\n\n")
        batch_idx += 1
        time.sleep(5)
