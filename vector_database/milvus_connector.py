from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility, BulkInsertState
from minio import Minio
from minio.error import S3Error
from typing import List, Optional
import os
import time


class MilvusClient:
    def __init__(self, host: str = "localhost", port: str = "19530", database: str = "Knowledge1024Hybrid",
                 collection_name: str = "telecom_dag_index_news", dim: int = 768,
                 minio_host: str = "localhost", minio_port: str = "9000", minio_access_key: str = "minioadmin",
                 minio_secret_key: str = "minioadmin",
                 minio_bucket: str = "milvus-bucket", remote_data_path: str = "milvus_bulk_data"
                 ):
        """
        初始化Milvus连接
        :param host: Milvus服务的地址
        :param port: Milvus服务的端口
        :param collection_name: 要使用的集合名称
        :param dim: 向量维度
        """

        self.collection_name = collection_name
        self.dim = dim
        connections.connect(host=host, port=port, db_name=database)
        self._create_collection_if_not_exists()

        # connect minio
        self.minio_address = f"{minio_host}:{minio_port}"
        self.minio_access_key = minio_access_key
        self.minio_secret_key = minio_secret_key
        self.minio_bucket = minio_bucket
        self.remote_data_path = remote_data_path
        self.minio_client = self._connect_minio()

    def _create_collection_if_not_exists(self):
        if self.collection_name in Collection.list_collections():
            self.collection = Collection(self.collection_name)
            return

        fields = [
            FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name='qa_id', dtype=DataType.INT64, is_primary=False, auto_id=False),
            FieldSchema(name='question', dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name='answer', dtype=DataType.VARCHAR, max_length=20000),
            FieldSchema(name='file_id', dtype=DataType.INT64, is_primary=False),
            FieldSchema(name='block_id', dtype=DataType.INT64, is_primary=False),
            FieldSchema(name='file_name', dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name='content', dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name='dense_embedding', dtype=DataType.FLOAT_VECTOR, dim=1024),
            FieldSchema(name='sparse_embedding', dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name='source', dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name='flag', dtype=DataType.VARCHAR, max_length=100)
        ]
        schema = CollectionSchema(fields, description="Collection for text embeddings")
        self.collection = Collection(name=self.collection_name, schema=schema)
        print(f"Created collection: {self.collection_name}")

    def _connect_minio(self):
        client = Minio(
            endpoint=self.minio_address,
            access_key=self.minio_access_key,
            secret_key=self.minio_secret_key,
            secure=False
        )
        if not client.bucket_exists(self.minio_bucket):
            client.make_bucket(self.minio_bucket)
            print(f"[INFO] Created MinIO bucket: {self.minio_bucket}")
        return client

    def ping(self) -> bool:
        try:
            return utility.has_collection(self.collection_name)
        except Exception:
            return False

    def insert(self, data: List[List]):
        """
        批量导入数据
        """
        if len(data[0]) == 0:
            raise ValueError(f"[ERROR] No data found!")

        self.collection.insert(data)
        self.collection.flush()
        print(f"[INFO] Inserted {len(data[0])} embedding.")

    def bulk_insert(self, parquet_file_path: str) -> bool:
        """
        上传Parquet文件到MinIO并触发Milvus的Bulk Insert
        :param parquet_file_path: 本地Parquet文件路径
        :return: 是否成功
        """
        # 1. 上传Parquet文件到 MinIO
        try:
            file_name = os.path.basename(parquet_file_path)
            remote_file_path = os.path.join(self.remote_data_path, "parquet", file_name)
            self.minio_client.fput_object(self.minio_bucket, remote_file_path, parquet_file_path)
            print(f"[INFO] Uploaded file '{file_name}' to MinIO at '{remote_file_path}'")
        except S3Error as e:
            print(f"[ERROR] Failed to upload file to MinIO: {e}")
            return False

        # 2. 发起Milvus的bulk insert
        try:
            task_id = utility.do_bulk_insert(
                collection_name=self.collection_name,
                files=[remote_file_path]
            )
            print(f"[INFO] Started BulkInsert, task_id: {task_id}")
        except Exception as e:
            print(f"[ERROR] Failed to start BulkInsert: {e}")
            return False

        # 3. 等待导入完成
        return self._wait_bulk_insert_complete(task_id)

    def check_exists(self, file_id: int) -> Optional[str]:
        """
        检查某个file_id是否存在
        :return: 存在返回file_name，不存在返回None
        """
        expr = f"file_id == {file_id}"
        results = self.collection.query(expr, output_fields=["file_name"])

        if results:
            return results[0]['file_name']
        return None

    def search(self, query_vectors: List[List[float]], top_k: int = 5, search_params: Optional[dict] = None):
        if search_params is None:
            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

        results = self.collection.search(
            data=query_vectors,
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["file_id", "file_name"]
        )
        return results

    def delete(self, expr: str):
        self.collection.delete(expr)
        self.collection.flush()
        print(f"[INFO] Deleted entries where {expr}")

    def drop_collection(self):
        self.collection.drop()
        print(f"[INFO] Dropped collection: {self.collection_name}")

    def _wait_bulk_insert_complete(self, task_id: int, timeout_sec: int = 600) -> bool:
        """
        等待BulkInsert完成
        """
        start_time = time.time()
        while True:
            states = utility.get_bulk_insert_state(task_id)
            if states.state == BulkInsertState.ImportCompleted:
                print(f"[SUCCESS] Bulk insert completed: {task_id}")
                return True
            elif states.state in (BulkInsertState.ImportFailed, BulkInsertState.ImportFailedAndCleaned):
                print(f"[FAILED] Bulk insert failed: {task_id}, reason: {states.failed_reason}")
                return False
            else:
                if time.time() - start_time > timeout_sec:
                    print(f"[TIMEOUT] Bulk insert timeout after {timeout_sec}s")
                    return False
                time.sleep(5)