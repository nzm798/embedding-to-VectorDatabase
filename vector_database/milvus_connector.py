from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility, BulkInsertState
from minio import Minio
from minio.error import S3Error
from typing import List, Optional
import os
import time


class MilvusClient:
    def __init__(self, host: str = "127.0.0.1", port: str = "19530", database: str = "Knowledge1024Hybrid",
                 collection_name: str = "telecom_dag_index_news", dim: int = 768,
                 minio_host: str = "127.0.0.1", minio_port: str = "9000", minio_access_key: str = "minioadmin",
                 minio_secret_key: str = "minioadmin",
                 minio_bucket: str = "news-bucket", remote_data_path: str = "./"
                 ):
        """
        初始化Milvus连接
        :param host: Milvus服务的地址
        :param port: Milvus服务的端口
        :param collection_name: 要使用的集合名称
        :param dim: 向量维度
        """

        self.collection_name = collection_name
        self.database = database
        self.dim = dim
        connections.connect(db_name=database, host=host, port=port)
        try:
            version = utility.get_server_version()
            print(f"[INFO] Successfully connected to Milvus. Server version: {version}")
        except Exception as e:
            raise Exception(f"[ERROR] Failed to verify Milvus connection: {e}")
        self._create_collection_if_not_exists()

        # connect minio
        self.minio_address = f"{minio_host}:{minio_port}"
        self.minio_access_key = minio_access_key
        self.minio_secret_key = minio_secret_key
        self.minio_bucket = minio_bucket
        self.remote_data_path = remote_data_path
        self.minio_client = self._connect_minio()

    def _create_collection_if_not_exists(self):
        if utility.has_collection(collection_name=self.collection_name):
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

        dense_index = {
            'metric_type': 'IP',
            'index_type': "FLAT",
            'params': {"nlist": 128}
        }

        sparse_index = {
            'metric_type': 'IP',
            'index_type': "SPARSE_INVERTED_INDEX",
        }
        if not utility.has_collection(self.collection_name):
            schema = CollectionSchema(fields=fields)
            collection = Collection(self.collection_name, schema=schema)
            collection.create_index(field_name="dense_embedding", index_params=dense_index)
            collection.create_index(field_name="sparse_embedding", index_params=sparse_index)
            collection.load()
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


if __name__ == '__main__':
    import random

    client = MilvusClient(
        host="192.168.0.110",
        port="19530",
        database="Knowledge1024Hybrid",
        minio_host="192.168.0.110",
        minio_port="9000",
        minio_access_key="minioadmin",
        minio_secret_key="minioadmin",
        minio_bucket="news-bucket",
        remote_data_path="./"
    )

    # 2. 测试连接
    assert client.ping(), "[FAIL] Cannot connect to Milvus!"

    # 3. 插入数据
    n = 5

    # dense_vectors：每个向量1024维
    dense_vectors = [[random.random() for _ in range(1024)] for _ in range(n)]

    sparse_vectors = []
    for _ in range(n):
        num_nonzero = random.randint(3, 10)  # 每个稀疏向量3~10个非零值
        indices = random.sample(range(1024), num_nonzero)  # 随机选索引
        indices.sort()  # 按索引排序（某些版本的Milvus可能要求索引有序）
        values = [random.random() for _ in range(num_nonzero)]  # 随机生成权重

        # 使用正确的稀疏向量格式: {indices: [...], values: [...]}
        sparse_vectors.append({"indices": indices, "values": values})

    # 准备要插入的数据
    data = [
        [0] * n,  # qa_id
        [""] * n,  # question
        [""] * n,  # answer
        [i for i in range(n)],  # file_id
        [i for i in range(n)],  # block_id
        [f"file_{i}.txt" for i in range(n)],  # file_name
        [f"content of file {i}" for i in range(n)],  # content
        dense_vectors,  # dense_embedding
        sparse_vectors,  # sparse_embedding
        [""] * n,  # source
        ["0"] * n  # flag
    ]

    # 插入
    client.insert(data)
    print("[PASS] Inserted test data.")
