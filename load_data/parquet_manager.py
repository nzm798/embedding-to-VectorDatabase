import os
import time
import pandas as pd
import threading
import logging
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Union, Any
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("./logs/parquet_manager.log"),
        logging.StreamHandler()
    ]
)


@dataclass
class ParquetFileInfo:
    """存储Parquet文件相关信息的数据类"""
    file_name: str
    file_path: str
    record_count: int = 0
    min_id: int = None
    max_id: int = None
    file_size_bytes: int = 0
    is_full: bool = False
    created_at: str = None
    last_updated_at: str = None
    data_buffer: Optional[pd.DataFrame] = None  # 添加数据缓冲区

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.last_updated_at is None:
            self.last_updated_at = self.created_at
        if self.data_buffer is None:
            self.data_buffer = pd.DataFrame()


class ParquetFileManager:
    """
    管理Parquet文件的创建、导入和删除的类
    """

    def __init__(self,
                 output_dir: str = "./parquet",
                 max_records_per_file: int = 100000,
                 max_file_size_mb: int = 1024,  # 1GB
                 max_files: int = 8,
                 max_return_files: int = 4,
                 log_file: str = "./logs/parquet_operations.log",
                 metadata_file: str = "./logs/parquet_metadata.json"):
        """
        初始化ParquetFileManager

        Args:
            output_dir: Parquet文件存储目录
            max_records_per_file: 单个文件最大记录数
            max_file_size_mb: 单个文件最大大小(MB)
            max_files: 最大允许的文件数量
            log_file: 操作日志文件路径
            metadata_file: 元数据保存文件路径
        """
        # 创建输出目录(如果不存在)
        os.makedirs(output_dir, exist_ok=True)
        # 创建日志目录(如果不存在)
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        self.output_dir = output_dir
        self.max_records_per_file = max_records_per_file
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.max_files = max_files
        self.log_file = log_file
        self.metadata_file = metadata_file
        self.max_return_files = max_return_files

        # 用于线程安全操作的锁
        self.lock = threading.RLock()  # 使用可重入锁

        # 文件信息字典: {file_name: ParquetFileInfo}
        self.file_info_table: Dict[str, ParquetFileInfo] = {}

        # 当前活跃文件
        self.current_file: Optional[ParquetFileInfo] = None

        # 文件句柄缓存，存储活跃文件的writer对象
        self.file_writers: Dict[str, pa.parquet.ParquetWriter] = {}

        # 文件写入锁，用于细粒度控制每个文件的写入
        self.file_locks: Dict[str, threading.RLock] = {}

        # 加载已存在的元数据
        self._load_metadata()

        # 初始化一个logger专门用于记录文件操作
        self.operation_logger = logging.getLogger('parquet_operations')
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.operation_logger.addHandler(file_handler)
        self.operation_logger.setLevel(logging.INFO)

        self.logger = logging.getLogger('ParquetFileManager')
        self.logger.info(
            f"初始化ParquetFileManager，输出目录: {output_dir}, 最大记录数: {max_records_per_file}, 最大文件大小: {max_file_size_mb}MB")

    def _load_metadata(self):
        """从元数据文件加载文件信息表"""
        with self.lock:
            if os.path.exists(self.metadata_file):
                try:
                    with open(self.metadata_file, 'r') as f:
                        metadata = json.load(f)

                    for file_info_dict in metadata:
                        # 移除data_buffer字段，因为JSON不能序列化DataFrame
                        if 'data_buffer' in file_info_dict:
                            del file_info_dict['data_buffer']

                        file_info = ParquetFileInfo(**file_info_dict)
                        self.file_info_table[file_info.file_name] = file_info

                        # 为每个文件创建对应的锁
                        self.file_locks[file_info.file_name] = threading.RLock()

                        # 找到最近的非满文件作为当前文件
                        if not file_info.is_full and (self.current_file is None or
                                                      file_info.last_updated_at > self.current_file.last_updated_at):
                            self.current_file = file_info

                    self.logger.info(f"已加载{len(self.file_info_table)}个文件的元数据")
                except Exception as e:
                    self.logger.error(f"加载元数据失败: {e}")
            else:
                self.logger.info("未找到元数据文件，将创建新的元数据")

    def _save_metadata(self):
        """保存文件信息表到元数据文件"""
        with self.lock:
            try:
                # 创建一个没有data_buffer字段的副本
                metadata_to_save = []
                for info in self.file_info_table.values():
                    info_dict = asdict(info)
                    if 'data_buffer' in info_dict:
                        del info_dict['data_buffer']
                    metadata_to_save.append(info_dict)

                with open(self.metadata_file, 'w') as f:
                    json.dump(metadata_to_save, f, indent=2)
                self.logger.debug("元数据已保存")
            except Exception as e:
                self.logger.error(f"保存元数据失败: {e}")

    def _get_writer(self, file_info: ParquetFileInfo, schema=None) -> pa.parquet.ParquetWriter:
        """
        获取或创建文件的ParquetWriter

        Args:
            file_info: 文件信息
            schema: 可选的Arrow模式，用于新文件

        Returns:
            ParquetWriter对象
        """
        file_lock = self.file_locks.get(file_info.file_name)
        if not file_lock:
            with self.lock:
                if file_info.file_name not in self.file_locks:
                    self.file_locks[file_info.file_name] = threading.RLock()
                file_lock = self.file_locks[file_info.file_name]

        with file_lock:
            if file_info.file_name in self.file_writers:
                return self.file_writers[file_info.file_name]

            # 检查文件是否已存在
            file_exists = os.path.exists(file_info.file_path)

            if file_exists:
                # 读取原有文件的模式
                existing_schema = pq.read_schema(file_info.file_path)
                # 创建一个可以附加到现有文件的写入器
                writer = pq.ParquetWriter(file_info.file_path, existing_schema, append=True)
            else:
                # 如果没有提供模式且文件不存在，无法创建写入器
                if schema is None:
                    raise ValueError(f"无法为新文件创建写入器: {file_info.file_name}, 因为没有提供模式")
                writer = pq.ParquetWriter(file_info.file_path, schema)

            # 缓存写入器
            with self.lock:
                self.file_writers[file_info.file_name] = writer
            return writer

    def _close_writer(self, file_name: str):
        """
        关闭并移除指定文件的writer

        Args:
            file_name: 文件名
        """
        file_lock = self.file_locks.get(file_name)
        if file_lock:
            with file_lock:
                with self.lock:
                    if file_name in self.file_writers:
                        try:
                            self.file_writers[file_name].close()
                        except Exception as e:
                            self.logger.error(f"关闭文件writer失败: {file_name}, 错误: {e}")
                        finally:
                            del self.file_writers[file_name]

    def _close_all_writers(self):
        """关闭所有打开的writer"""
        with self.lock:
            for file_name, writer in list(self.file_writers.items()):
                self._close_writer(file_name)

    def _generate_file_name(self) -> str:
        """生成新文件名，格式: data_YYYYMMDD_HHMMSS.parquet"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"data_{timestamp}.parquet"

    def _get_or_create_current_file(self) -> Tuple[ParquetFileInfo, bool]:
        """获取或创建当前文件，返回(文件信息, 是否新创建)"""
        with self.lock:
            # 如果已有非满文件，直接使用
            if self.current_file and not self.current_file.is_full:
                return self.current_file, False

            # 检查是否达到最大文件数量
            if len([info for info in self.file_info_table.values() if not info.is_full]) >= self.max_files:
                self.logger.warning(f"已达到最大文件数量限制({self.max_files})，无法创建新文件")
                return None, False

            # 创建新文件
            file_name = self._generate_file_name()
            file_path = os.path.join(self.output_dir, file_name)

            new_file_info = ParquetFileInfo(
                file_name=file_name,
                file_path=file_path,
                created_at=datetime.now().isoformat(),
                last_updated_at=datetime.now().isoformat()
            )

            self.file_info_table[file_name] = new_file_info
            self.current_file = new_file_info

            # 为新文件创建锁
            self.file_locks[file_name] = threading.RLock()

            self._save_metadata()
            self.logger.info(f"创建新文件: {file_name}")
            return new_file_info, True

    def _convert_to_dataframe(self, data: Any) -> pd.DataFrame:
        """
        将各种数据格式转换为DataFrame

        支持以下格式:
        1. 已有的DataFrame
        2. 字典列表 [{key: value}, ...]
        3. 字典 {key: value}
        4. 列数据字典 {"col1": [val1, val2, ...], "col2": [val1, val2, ...]}

        Args:
            data: 要转换的数据

        Returns:
            转换后的DataFrame
        """
        if isinstance(data, pd.DataFrame):
            return data

        if isinstance(data, dict):
            # 检查是否是"列数据字典"格式
            if all(isinstance(v, list) for v in data.values()):
                # 检查所有列的长度是否一致
                lengths = [len(v) for v in data.values()]
                if len(set(lengths)) == 1:  # 所有列长度相同
                    return pd.DataFrame(data)
                else:
                    self.logger.warning("列数据字典中的各列长度不一致，将尝试自动调整")
                    # 尝试调整不一致的列长度
                    max_len = max(lengths)
                    adjusted_data = {}
                    for k, v in data.items():
                        if len(v) < max_len:
                            # 填充缺失值
                            adjusted_data[k] = v + [None] * (max_len - len(v))
                        else:
                            adjusted_data[k] = v
                    return pd.DataFrame(adjusted_data)
            else:
                # 普通字典，转换为单行DataFrame
                return pd.DataFrame([data])

        if isinstance(data, list):
            if all(isinstance(item, dict) for item in data):
                # 字典列表
                return pd.DataFrame(data)
            else:
                # 其他列表类型，尝试转换
                self.logger.warning("无法识别的列表数据格式，尝试转换")
                return pd.DataFrame(data)

        raise ValueError(f"无法识别的数据格式: {type(data)}")

    def write_data(self, data: Any, id_column: str = 'file_id') -> bool:
        """
        将数据写入Parquet文件 - 线程安全版本

        Args:
            data: 要写入的数据，支持多种格式
            id_column: ID列名，用于追踪最小ID和最大ID

        Returns:
            是否成功写入
        """
        try:
            # 将输入数据转换为DataFrame（不需要锁保护）
            df = self._convert_to_dataframe(data)

            if df.empty:
                self.logger.warning("尝试写入空数据，已忽略")
                return False

            # 获取或创建当前文件（需要锁保护）
            with self.lock:
                file_info, is_new = self._get_or_create_current_file()
                if file_info is None:
                    return False

                # 获取文件对应的锁
                file_lock = self.file_locks[file_info.file_name]

            # 对特定文件的操作使用文件锁
            with file_lock:
                try:
                    # 准备数据和schema
                    record_count = len(df)
                    table = pa.Table.from_pandas(df)
                    schema = table.schema

                    # 获取writer（带文件锁的方法调用）
                    writer = self._get_writer(file_info, schema)

                    # 写入数据
                    writer.write_table(table)

                    # 更新文件统计信息（需要文件锁保护）
                    file_size = os.path.getsize(file_info.file_path)
                    current_time = datetime.now().isoformat()

                    # 更新最小和最大ID
                    min_id = None
                    max_id = None
                    if id_column in df.columns:
                        min_id = df[id_column].min()
                        max_id = df[id_column].max()

                    # 检查文件是否已满
                    is_full = False
                    if ((file_info.record_count + record_count) >= self.max_records_per_file or
                            file_size >= self.max_file_size_bytes):
                        is_full = True

                    # 获取全局锁更新元数据
                    with self.lock:
                        # 更新文件信息
                        file_info.record_count += record_count
                        file_info.file_size_bytes = file_size
                        file_info.last_updated_at = current_time

                        # 更新ID范围
                        if id_column in df.columns:
                            if file_info.min_id is None or min_id < file_info.min_id:
                                file_info.min_id = int(min_id)
                            if file_info.max_id is None or max_id > file_info.max_id:
                                file_info.max_id = int(max_id)

                        # 处理文件已满的情况
                        if is_full:
                            file_info.is_full = True
                            # 如果这是当前文件，则重置当前文件
                            if self.current_file and self.current_file.file_name == file_info.file_name:
                                self.current_file = None

                            # 关闭writer（不在这里调用，因为会导致死锁）
                            # self._close_writer(file_info.file_name)
                            self.logger.info(f"文件已满: {file_info.file_name}, 记录数: {file_info.record_count}, "
                                             f"大小: {file_info.file_size_bytes / 1024 / 1024:.2f}MB")

                        # 保存元数据
                        self._save_metadata()

                    # 如果文件已满，关闭写入器（已经有文件锁保护，不会导致死锁）
                    if is_full:
                        if file_info.file_name in self.file_writers:
                            try:
                                self.file_writers[file_info.file_name].close()
                            except Exception as e:
                                self.logger.error(f"关闭文件writer失败: {file_info.file_name}, 错误: {e}")

                            # 获取全局锁删除writer引用
                            with self.lock:
                                if file_info.file_name in self.file_writers:
                                    del self.file_writers[file_info.file_name]

                    self.logger.debug(f"成功写入{record_count}条记录到文件: {file_info.file_name}")
                    return True

                except Exception as e:
                    # 处理异常，关闭writer
                    self.logger.error(f"写入数据到文件{file_info.file_name}失败: {e}")

                    # 尝试关闭writer（已经有文件锁保护）
                    if file_info.file_name in self.file_writers:
                        try:
                            self.file_writers[file_info.file_name].close()
                        except Exception as close_error:
                            self.logger.error(f"关闭writer失败: {close_error}")

                        # 获取全局锁删除writer引用
                        with self.lock:
                            if file_info.file_name in self.file_writers:
                                del self.file_writers[file_info.file_name]

                    return False

        except Exception as e:
            self.logger.error(f"处理数据失败: {e}")
            return False

    def write_columns_data(self, columns_data: Dict[str, List], id_column: str = 'file_id') -> bool:
        """
        将列数据格式写入Parquet文件

        Args:
            columns_data: 列数据字典，格式为 {"col1": [val1, val2, ...], "col2": [val1, val2, ...]}
            id_column: ID列名，用于追踪最小ID和最大ID

        Returns:
            是否成功写入
        """
        return self.write_data(columns_data, id_column)

    def get_full_files(self, is_finally: bool = False) -> List[ParquetFileInfo]:
        """
        获取所有已写满的文件信息

        Returns:
            已写满文件的信息列表
        """
        infos=[]
        with self.lock:
            if is_finally:
                infos = [info for info in self.file_info_table.values()]
            infos = [info for info in self.file_info_table.values() if info.is_full]
            if len(infos) > self.max_return_files:
                infos = infos[:self.max_return_files]
            return infos
    def process_full_files(self, is_finally: bool = False) -> List[ParquetFileInfo]:
        """
        处理已写满的文件：记录日志并从表中移除

        Returns:
            处理的文件信息列表
        """
        with self.lock:
            full_files = self.get_full_files(is_finally)
            if not full_files:
                self.logger.info("没有写满的文件需要处理")
                return []

            # 记录到操作日志
            for file_info in full_files:
                # 确保文件写入器已关闭
                self._close_writer(file_info.file_name)

                log_message = (
                    f"FILE_PROCESSED|{file_info.file_name}|{file_info.file_path}|"
                    f"{file_info.record_count}|{file_info.min_id}|{file_info.max_id}|"
                    f"{file_info.file_size_bytes}|{file_info.created_at}|{file_info.last_updated_at}"
                )
                self.operation_logger.info(log_message)

                # 从表中移除
                self.file_info_table.pop(file_info.file_name, None)

                # 移除对应的文件锁
                if file_info.file_name in self.file_locks:
                    del self.file_locks[file_info.file_name]

            self._save_metadata()
            self.logger.info(f"已处理{len(full_files)}个写满的文件")
            return full_files

    def get_file_info(self, file_name: str = None) -> Union[ParquetFileInfo, List[ParquetFileInfo], None]:
        """
        获取文件信息

        Args:
            file_name: 要获取的文件名，如果为None则返回所有文件信息

        Returns:
            文件信息或文件信息列表
        """
        with self.lock:
            if file_name is None:
                return list(self.file_info_table.values())
            return self.file_info_table.get(file_name)

    def get_stats(self) -> Dict:
        """获取文件管理器状态统计"""
        with self.lock:
            total_records = sum(info.record_count for info in self.file_info_table.values())
            total_size_mb = sum(info.file_size_bytes for info in self.file_info_table.values()) / (1024 * 1024)
            full_files = sum(1 for info in self.file_info_table.values() if info.is_full)
            active_files = sum(1 for info in self.file_info_table.values() if not info.is_full)
            active_writers = len(self.file_writers)

            return {
                "total_files": len(self.file_info_table),
                "full_files": full_files,
                "active_files": active_files,
                "active_writers": active_writers,
                "total_records": total_records,
                "total_size_mb": round(total_size_mb, 2),
                "max_files_limit": self.max_files,
                "can_create_new_file": active_files < self.max_files
            }

    def close(self):
        """关闭所有打开的文件，保存元数据"""
        self._close_all_writers()
        with self.lock:
            self._save_metadata()
        self.logger.info("ParquetFileManager已关闭所有文件")

    def __del__(self):
        """析构函数，确保所有文件被关闭，元数据被保存"""
        try:
            self.close()
        except Exception as e:
            # 在析构函数中避免抛出异常
            pass
