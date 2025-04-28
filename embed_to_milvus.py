import json
import threading
from concurrent.futures import ThreadPoolExecutor
from load_data.json_batch_reader import JsonlBatchReader
from load_data.parquet_manager import ParquetFileManager
from embedding_model.tei_req import TeiEmbeddingClient
from splite_text.lang_chain_splitter import TextSplitter
from vector_database.milvus_connector import MilvusClient
from vector_database.mysql_connector import MySQLClient


class EmbeddToMilvus:
    def __init__(
            self,
            reader: JsonlBatchReader,
            parquet_manager: ParquetFileManager,
            embedding_client: TeiEmbeddingClient,
            milvus_client: MilvusClient,
            mysql_client: MySQLClient,
            text_splitter: TextSplitter,
            processing_thread_num: int = 4,
            upload_thread_num: int = 2,
            max_pending_files: int = 10
    ):
        self.reader = reader
        self.parquet_manager = parquet_manager
        self.embedding_client = embedding_client
        self.milvus_client = milvus_client
        self.mysql_client = mysql_client
        self.text_splitter = text_splitter

        # 线程数设置
        self.processing_thread_num = processing_thread_num
        self.upload_thread_num = upload_thread_num

        # 用于跟踪处理状态的标志和锁
        self.is_processing_complete = False
        self.pending_files_count = 0
        self.max_pending_files = max_pending_files

        # 锁和条件变量
        self.file_count_lock = threading.Lock()
        self.file_available_condition = threading.Condition(self.file_count_lock)
        self.file_space_condition = threading.Condition(self.file_count_lock)

    def process_batch_and_write(self):
        """
        从reader获取批次，处理embedding并写入parquet
        多线程同时执行这个函数
        """
        while True:
            # 获取批次数据
            batch = None
            if self.reader.has_next_batch():
                batch = self.reader.next_batch()

            # 如果没有更多批次，结束处理
            if batch is None:
                break

            try:
                embedding_texts = []
                file_ids = []
                block_ids = []
                # 提取文章标题和内容
                titles = batch["title"]
                pub_times = batch["pub_time"]
                contents = batch["content"]
                sources = batch["source"]
                file_names = batch["file_names"]

                batch_num = len(batch["title"])

                for i in range(batch_num):
                    file_id = self.mysql_client.get_id_by_filename(file_names[i])
                    if not file_id:
                        continue
                    is_exist=self.milvus_client.check_exists(file_id)
                    if not is_exist:
                        print(f"[INFO] {is_exist} is existed")
                        continue
                    text = f"[标题]:{titles[i]}\n[时间]:{pub_times[i]}\n[来源]:{sources[i]}\n\n{contents[i]}"
                    blocks = self.text_splitter.split(text)
                    if blocks:
                        for block_id, content in enumerate(blocks):
                            embedding_texts.append(content)
                            file_ids.append(file_id)
                            block_ids.append(block_id)

                dense_embeddings, sparse_embeddings = self.embedding_client.embed_all(embedding_texts)

                columns_data = {
                    'qa_id':[0] * batch_num,  # 占位符ID
                    'question':[""] * batch_num,  # 占位符1
                    'answer':[""] * batch_num,  # 占位符2
                    'file_id':file_ids,
                    'block_id':block_ids,
                    'file_name':file_names,
                    'content':embedding_texts,
                    'dense_embedding':dense_embeddings,
                    'sparse_embedding':sparse_embeddings,
                    'source':[""] * batch_num,  # 占位符3
                    'flag':["0"] * batch_num  # 占位符4
                }


                # 等待，直到有足够的空间写入新文件
                with self.file_count_lock:
                    while self.pending_files_count >= self.max_pending_files:
                        self.file_space_condition.wait()

                # 写入parquet文件
                success = self.parquet_manager.write_columns_data(columns_data, id_column="file_id")

                # 检查是否有新的完整文件可以上传
                self.check_and_signal_files()

            except Exception as e:
                print(f"Error processing batch: {e}")

    def check_and_signal_files(self, is_finally: bool = False):
        """检查并通知有新文件可以上传"""
        with self.file_count_lock:
            files = self.parquet_manager.get_full_files(is_finally=is_finally)
            if files:
                # 增加待处理文件计数
                self.pending_files_count += len(files)
                # 通知上传线程有新文件可用
                self.file_available_condition.notify_all()

    def upload_files(self):
        """上传文件到Milvus的工作线程"""
        while True:
            files_to_upload = []

            # 获取可用的文件
            with self.file_count_lock:
                while self.pending_files_count == 0:
                    if self.is_processing_complete:
                        return  # 如果处理已完成且没有文件，则退出
                    self.file_available_condition.wait(timeout=1)
                    if self.pending_files_count == 0 and not self.is_processing_complete:
                        continue

                # 获取文件并更新计数
                files = self.parquet_manager.process_full_files(is_finally=self.is_processing_complete)
                if files:
                    files_to_upload = files
                    self.pending_files_count -= len(files)
                    # 通知有空间可以写入新文件
                    self.file_space_condition.notify_all()
            # 执行上传
            for file_info in files_to_upload:
                try:
                    self.milvus_client.bulk_insert(file_info.file_path)
                except Exception as e:
                    print(f"Error uploading file: {e}")

    def run(self):
        """启动整个处理流程"""
        processing_threads = []
        for _ in range(self.processing_thread_num):
            thread = threading.Thread(target=self.process_batch_and_write)
            thread.daemon = True
            processing_threads.append(thread)
            thread.start()

        # 启动上传线程池
        upload_executor = ThreadPoolExecutor(max_workers=self.upload_thread_num)
        upload_futures = []
        for _ in range(self.upload_thread_num):
            future = upload_executor.submit(self.upload_files)
            upload_futures.append(future)

        # 等待所有处理线程完成
        for thread in processing_threads:
            thread.join()

        # 标记处理完成
        self.is_processing_complete = True

        # 处理最后剩余的文件
        self.check_and_signal_files(is_finally=True)

        # 通知上传线程处理已完成
        with self.file_count_lock:
            self.file_available_condition.notify_all()

        # 关闭上传线程池
        upload_executor.shutdown(wait=True)

        print("Data processing complete!")


def main():
    # 加载配置文件
    with open("config.json", "r") as config_file:
        config = json.load(config_file)

    # 初始化组件
    # 从配置文件加载相应的参数
    reader_config = config["BatchReader"]
    reader = JsonlBatchReader(
        file_path=reader_config["file_path"],
        start=reader_config["start"],
        end=reader_config["end"],
        batch_size=reader_config["batch_size"]
    )

    parquet_config = config["ParquetFile"]
    parquet_manager = ParquetFileManager(
        output_dir=parquet_config["output_dir"],
        max_records_per_file=parquet_config["max_records_per_file"],
        max_file_size_mb=parquet_config["max_file_size_mb"],
        max_files=parquet_config["max_files"],
        max_return_files=parquet_config["max_return_files"],
        log_file=parquet_config["log_file"],
        metadata_file=parquet_config["metadata_file"]
    )

    embedding_config = config["TeiEmbed"]
    embedding_client = TeiEmbeddingClient(
        api_host=embedding_config["host"],
        api_port=embedding_config["port"],
        api_key=embedding_config["key"]
    )

    milvus_config = config["Milvus"]
    milvus_client = MilvusClient(
        host=milvus_config["host"],
        port=milvus_config["port"],
        database=milvus_config["database"],
        collection_name=milvus_config["collection_name"],
        minio_host=milvus_config["minio_host"],
        minio_port=milvus_config["minio_port"],
        minio_access_key=milvus_config["minio_access_key"],
        minio_secret_key=milvus_config["minio_secret_key"],
        minio_bucket=milvus_config["minio_bucket"],
        remote_data_path=milvus_config["remote_data_path"]
    )

    mysql_config = config["Mysql"]
    mysql_client = MySQLClient(
        host=mysql_config["host"],
        port=mysql_config["port"],
        user=mysql_config["user"],
        password=mysql_config["password"],
        database=mysql_config["database"]
    )

    splitter_config = config["Splitter"]
    text_splitter = TextSplitter(
        chunk_size=splitter_config["chunk_size"],
        overlap=splitter_config["overlap"]
    )

    # 创建并运行处理器
    processor = EmbeddToMilvus(
        reader=reader,
        parquet_manager=parquet_manager,
        embedding_client=embedding_client,
        milvus_client=milvus_client,
        mysql_client=mysql_client,
        text_splitter=text_splitter,
        processing_thread_num=4,  # 处理数据的线程数
        upload_thread_num=2,  # 上传到Milvus的线程数
        max_pending_files=10  # 最大待处理文件数量
    )

    # 启动数据处理与上传流程
    processor.run()


if __name__ == "__main__":
    main()
