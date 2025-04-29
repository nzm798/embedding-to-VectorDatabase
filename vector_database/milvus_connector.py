from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility, BulkInsertState, \
    MilvusClient
from minio import Minio
from minio.error import S3Error
from typing import List, Optional
import os
import time


class MyMilvusClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 19530, database: str = "Knowledge1024Hybrid",
                 collection_name: str = "telecom_dag_index_news", dim: int = 768,
                 minio_host: str = "127.0.0.1", minio_port: int = 9000, minio_access_key: str = "minioadmin",
                 minio_secret_key: str = "minioadmin",
                 minio_bucket: str = "a-bucket", remote_data_path: str = "parquet"
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

    def bulk_insert(self, path_name, parquet_file_path: List[List[str]]) -> bool:
        """
        上传Parquet文件到MinIO并触发Milvus的Bulk Insert
        :param parquet_file_path: 本地Parquet文件路径
        :param path_name: 该批次数据的文件名
        :return: 是否成功
        """
        # 1. 上传所有Parquet文件到 MinIO
        remote_file_paths = []  # 存储上传到 MinIO 的文件路径
        try:
            for path in parquet_file_path:  # 遍历嵌套列表中的每一个子列表
                file_name = os.path.basename(path[0])
                remote_file_path = os.path.join(self.remote_data_path, path_name, file_name).replace("\\", "/")
                # 上传文件到 MinIO
                self.minio_client.fput_object(self.minio_bucket, remote_file_path, path[0])
                remote_file_paths.append(remote_file_path)  # 记录已上传的文件路径
                print(f"[INFO] Uploaded file '{file_name}' to MinIO at '{remote_file_path}'")
        except S3Error as e:
            print(f"[ERROR] Failed to upload file to MinIO: {e}")
            return False

        # 2. 发起Milvus的bulk insert
        try:
            task_id = utility.do_bulk_insert(
                collection_name=self.collection_name,
                files=remote_file_paths
            )
            print(f"***********************"
                  f"[INFO] Bulk insert task ID: {task_id} \n"
                  f"This task uploads files:\n"
                  f"{remote_file_paths}"
                  f"***********************"
                  )

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

    # client = MyMilvusClient(
    #     host="192.168.0.110",
    #     port=19530,
    #     database="Knowledge1024Hybrid",
    #     minio_host="192.168.0.110",
    #     minio_port=9000,
    #     minio_access_key="minioadmin",
    #     minio_secret_key="minioadmin",
    #     minio_bucket="a-bucket",
    # )

    # 2. 测试连接
    # assert client.ping(), "[FAIL] Cannot connect to Milvus!"
    from pymilvus import MilvusClient, DataType
    from load_data.milvus_bulk_writer import MilvusBulkWriterManager

    schema = MilvusClient.create_schema(
        auto_id=True,
        enable_dynamic_field=True
    )
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="qa_id", datatype=DataType.INT64, is_primary=False, auto_id=False)
    schema.add_field(field_name="question", datatype=DataType.VARCHAR, max_length=2000)
    schema.add_field(field_name="answer", datatype=DataType.VARCHAR, max_length=20000)
    schema.add_field(field_name="file_id", datatype=DataType.INT64, is_primary=False)
    schema.add_field(field_name="block_id", datatype=DataType.INT64, is_primary=False)
    schema.add_field(field_name="file_name", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="dense_embedding", datatype=DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field(field_name="sparse_embedding", datatype=DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="flag", datatype=DataType.VARCHAR, max_length=100)
    schema.verify()

    # 3. 插入数据
    n = 100

    # dense_vectors：每个向量1024维
    dense_vectors = [[random.random() for _ in range(1024)] for _ in range(n)]

    sparse_vectors = [
        {
            364: 0.17531773447990417,
            418: 0.145879546621,
            630: 0.1101302548795,
            3172: 0.268978546412,
            5357: 0.254789645874,
            15483: 0.215479896454225454
        } for _ in range(n)
    ]

    # 准备要插入的数据
    data = {
        "qa_id": [0] * n,  # qa_id
        "question": [""] * n,  # question
        "answer": [""] * n,  # answer
        "file_id": [i for i in range(n)],  # file_id
        "block_id": [i for i in range(n)],  # block_id
        "file_name": [f"file_{i}.txt" for i in range(n)],  # file_name
        "content": [f"content of file {i}" for i in range(n)],  # content
        "dense_embedding": dense_vectors,  # dense_embedding
        "sparse_embedding": sparse_vectors,  # sparse_embedding
        "source": [""] * n,  # source
        "flag": ["0"] * n  # flag
    }

    parquet = MilvusBulkWriterManager(schema=schema)
    parquet.write_columns_data(columns_data=data)
    file = parquet.process_full_files(include_active=True)
    print(file[0].batch_file)
    # client.bulk_insert(file[0].file_path)
