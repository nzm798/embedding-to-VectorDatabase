import os
import time
import logging
import json
import threading
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime
import numpy as np
import pandas as pd

# Import Milvus-specific modules
from pymilvus.bulk_writer import LocalBulkWriter, BulkFileType
from pymilvus import MilvusClient, DataType


@dataclass
class BulkFileInfo:
    """Stores information about a bulk file"""
    file_name: str
    file_path: str
    batch_file: List = None
    record_count: int = 0
    min_id: int = None
    max_id: int = None
    is_full: bool = False
    created_at: str = None
    last_updated_at: str = None
    data_buffer: Optional[pd.DataFrame] = None  # Optional data buffer

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.last_updated_at is None:
            self.last_updated_at = self.created_at
        if self.data_buffer is None:
            self.data_buffer = pd.DataFrame()


class MilvusBulkWriterManager:
    """
    Manages creation, writing, and management of Milvus bulk files
    using the LocalBulkWriter interface.
    """

    def __init__(self,
                 schema,
                 output_dir: str = "parquet",
                 max_records_per_file: int = 100000,
                 segment_size_mb: int = 512,  # 512MB default segment size
                 max_files: int = 8,
                 max_return_files: int = 4,
                 log_file: str = "./logs/bulk_writer_operations.log",
                 metadata_file: str = "./logs/bulk_writer_metadata.json",
                 file_type: BulkFileType = BulkFileType.PARQUET):
        """
        Initialize the MilvusBulkWriterManager

        Args:
            schema: Milvus schema definition
            output_dir: Directory to store bulk files
            max_records_per_file: Max records per file before starting a new one
            segment_size_mb: Max segment size in MB
            max_files: Maximum number of active files
            max_return_files: Maximum number of files to return in one operation
            log_file: Path to operations log file
            metadata_file: Path to metadata storage file
            file_type: Type of bulk file (default: PARQUET)
        """
        # Create output directory if not exists
        os.makedirs(output_dir, exist_ok=True)
        # Create log directory if not exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        os.makedirs(os.path.dirname("./logs/bulk_writer.log"), exist_ok=True)

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("./logs/bulk_writer.log", encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('MilvusBulkWriterManager')

        # Store configuration
        self.schema = schema
        self.output_dir = output_dir
        self.max_records_per_file = max_records_per_file
        self.segment_size_bytes = segment_size_mb * 1024 * 1024
        self.max_files = max_files
        self.max_return_files = max_return_files
        self.log_file = log_file
        self.metadata_file = metadata_file
        self.file_type = file_type

        # Thread safety
        self.lock = threading.RLock()  # Reentrant lock

        # File info tracking
        self.file_info_table: Dict[str, BulkFileInfo] = {}

        # Current active writer
        self.current_writer: Optional[LocalBulkWriter] = None
        self.current_file: Optional[BulkFileInfo] = None

        # Track active writers
        self.file_writers: Dict[str, LocalBulkWriter] = {}

        # File-specific locks
        self.file_locks: Dict[str, threading.RLock] = {}

        # Load existing metadata if available
        self._load_metadata()

        # Initialize operations logger
        self.operation_logger = logging.getLogger('bulk_writer_operations')
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.operation_logger.addHandler(file_handler)
        self.operation_logger.setLevel(logging.INFO)

        self.logger.info(
            f"Initialized MilvusBulkWriterManager, output directory: {output_dir}, "
            f"max records: {max_records_per_file}, segment size: {segment_size_mb}MB"
        )

    def _load_metadata(self):
        """Load file information from metadata file"""
        with self.lock:
            if os.path.exists(self.metadata_file):
                try:
                    with open(self.metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)

                    for file_info_dict in metadata:
                        # Remove data_buffer field as it can't be serialized
                        if 'data_buffer' in file_info_dict:
                            del file_info_dict['data_buffer']

                        file_info = BulkFileInfo(**file_info_dict)
                        self.file_info_table[file_info.file_name] = file_info

                        # Create locks for each file
                        self.file_locks[file_info.file_name] = threading.RLock()

                        # Find the most recent non-full file to use as current file
                        if not file_info.is_full and (self.current_file is None or
                                                      file_info.last_updated_at > self.current_file.last_updated_at):
                            self.current_file = file_info

                    self.logger.info(f"Loaded metadata for {len(self.file_info_table)} files")
                except Exception as e:
                    self.logger.error(f"Failed to load metadata: {e}")
            else:
                self.logger.info("No metadata file found, will create new metadata")

    def _save_metadata(self):
        """Save file information to metadata file"""
        import builtins
        with self.lock:
            try:
                # Create a copy without data_buffer field
                metadata_to_save = []
                for info in self.file_info_table.values():
                    info_dict = asdict(info)
                    if 'data_buffer' in info_dict:
                        del info_dict['data_buffer']
                    metadata_to_save.append(info_dict)

                with builtins.open(self.metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata_to_save, f, indent=2)
                self.logger.debug("Metadata saved")
            except Exception as e:
                self.logger.error(f"Failed to save metadata: {e}")

    def _generate_file_name(self) -> str:
        """Generate a new file name: bulk_YYYYMMDD_HHMMSS"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"bulk_{timestamp}"

    def _get_or_create_current_writer(self) -> Tuple[LocalBulkWriter, BulkFileInfo, bool]:
        """
        Get or create current writer and file info

        Returns:
            Tuple of (writer, file_info, is_new)
        """
        with self.lock:
            # Use existing non-full file if available
            if self.current_file and not self.current_file.is_full and self.current_writer:
                return self.current_writer, self.current_file, False

            # Check if we've reached max file limit
            if len([info for info in self.file_info_table.values() if not info.is_full]) >= self.max_files:
                self.logger.warning(f"Reached maximum file limit ({self.max_files}), cannot create new file")
                return None, None, False

            # Create new file
            file_name = self._generate_file_name()
            file_path = os.path.join(self.output_dir, file_name)

            new_file_info = BulkFileInfo(
                file_name=file_name,
                file_path=file_path,
                created_at=datetime.now().isoformat(),
                last_updated_at=datetime.now().isoformat()
            )

            # Create a new bulk writer
            try:
                new_writer = LocalBulkWriter(
                    schema=self.schema,
                    local_path=file_path,
                    segment_size=self.segment_size_bytes,
                    file_type=self.file_type
                )

                # Store in our tracking structures
                self.file_info_table[file_name] = new_file_info
                self.file_writers[file_name] = new_writer
                self.file_locks[file_name] = threading.RLock()

                # Set as current
                self.current_file = new_file_info
                self.current_writer = new_writer

                self._save_metadata()
                self.logger.info(f"Created new bulk file: {file_name}")

                return new_writer, new_file_info, True

            except Exception as e:
                self.logger.error(f"Failed to create bulk writer: {e}")
                return None, None, False

    def _convert_to_row_format(self, data: Any) -> List[Dict]:
        """
        Convert various data formats to a list of row dictionaries
        suitable for Milvus bulk writer

        Args:
            data: Input data in various formats

        Returns:
            List of row dictionaries
        """
        if isinstance(data, pd.DataFrame):
            return data.to_dict('records')

        if isinstance(data, dict):
            # Check if it's column format {"col1": [val1, val2], "col2": [val1, val2]}
            if all(isinstance(v, list) for v in data.values()):
                # Check if all columns have the same length
                lengths = [len(v) for v in data.values()]
                if len(set(lengths)) == 1:  # All columns have the same length
                    # Convert to rows format
                    rows = []
                    for i in range(lengths[0]):
                        row = {k: v[i] for k, v in data.items()}
                        rows.append(row)
                    return rows
                else:
                    self.logger.warning("Columns have different lengths, will try to adjust")
                    # Adjust inconsistent column lengths
                    max_len = max(lengths)
                    adjusted_data = {}
                    for k, v in data.items():
                        if len(v) < max_len:
                            # Fill with None values
                            adjusted_data[k] = v + [None] * (max_len - len(v))
                        else:
                            adjusted_data[k] = v

                    # Convert to rows
                    rows = []
                    for i in range(max_len):
                        row = {k: v[i] for k, v in adjusted_data.items()}
                        rows.append(row)
                    return rows
            else:
                # Single dictionary - convert to a single row
                return [data]

        if isinstance(data, list):
            if all(isinstance(item, dict) for item in data):
                # Already in row format
                return data
            else:
                # Try to convert other list formats
                self.logger.warning("Unrecognized list data format, attempting conversion")
                return data

        raise ValueError(f"Unrecognized data format: {type(data)}")

    def write_data(self, data: Any, id_column: str = 'file_id') -> bool:
        """
        Write data to bulk file - thread-safe

        Args:
            data: Data to write, supports various formats
            id_column: ID column name, used to track min/max IDs

        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert input data to row format (no lock needed)
            rows = self._convert_to_row_format(data)

            if not rows:
                self.logger.warning("Attempted to write empty data, ignored")
                return False

            # Get or create current writer (lock needed)
            writer, file_info, is_new = self._get_or_create_current_writer()
            if writer is None or file_info is None:
                return False

            # Get file-specific lock
            file_lock = self.file_locks[file_info.file_name]

            # Process with file lock
            with file_lock:
                try:
                    # Write each row to the bulk writer
                    record_count = len(rows)
                    for row in rows:
                        writer.append_row(row)

                    # Commit if we've reached a batch size
                    if file_info.record_count + record_count >= self.max_records_per_file:
                        writer.commit()
                        file_info.batch_file = writer.batch_files

                    # Update file statistics
                    current_time = datetime.now().isoformat()

                    # Update min and max IDs if available
                    min_id = None
                    max_id = None
                    if all(id_column in row for row in rows):
                        min_id = min(row[id_column] for row in rows if id_column in row)
                        max_id = max(row[id_column] for row in rows if id_column in row)

                    # Check if file is full after this write
                    is_full = False

                    if (file_info.record_count + record_count) >= self.max_records_per_file:
                        is_full = True

                    # Update metadata with global lock
                    with self.lock:
                        # Update file info
                        file_info.record_count += record_count
                        file_info.last_updated_at = current_time

                        # Update ID range
                        if min_id is not None:
                            if file_info.min_id is None or min_id < file_info.min_id:
                                file_info.min_id = int(min_id)
                        if max_id is not None:
                            if file_info.max_id is None or max_id > file_info.max_id:
                                file_info.max_id = int(max_id)

                        # Handle full file case
                        if is_full:
                            file_info.is_full = True

                            # Close writer with final commit
                            writer.commit()
                            # save finally file path
                            file_info.batch_file = writer.batch_files

                            # If this is current writer, reset
                            if self.current_file and self.current_file.file_name == file_info.file_name:
                                self.current_file = None
                                self.current_writer = None

                            self.logger.info(f"File is full: {file_info.file_name}, record count: {file_info.record_count}")

                        # Save metadata
                        self._save_metadata()

                    # If file is full, remove writer reference
                    if is_full:
                        with self.lock:
                            if file_info.file_name in self.file_writers:
                                del self.file_writers[file_info.file_name]

                    self.logger.debug(f"Successfully wrote {record_count} records to file: {file_info.file_name}")
                    return True

                except Exception as e:
                    self.logger.error(f"Failed to write data to file {file_info.file_name}: {e}")
                    return False

        except Exception as e:
            self.logger.error(f"Failed to process data: {e}")
            return False

    def write_columns_data(self, columns_data: Dict[str, List], id_column: str = 'file_id') -> bool:
        """
        Write column-formatted data to bulk file

        Args:
            columns_data: Column data in format {"col1": [val1, val2, ...], "col2": [val1, val2, ...]}
            id_column: ID column name

        Returns:
            True if successful, False otherwise
        """
        return self.write_data(columns_data, id_column)

    def get_full_files(self, include_active: bool = False) -> List[BulkFileInfo]:
        """
        Get information about all full files

        Args:
            include_active: Whether to include active (non-full) files

        Returns:
            List of file information objects
        """
        with self.lock:
            infos = []
            if include_active:
                infos = [info for info in self.file_info_table.values()]
            else:
                infos = [info for info in self.file_info_table.values() if info.is_full]

            if len(infos) > self.max_return_files:
                infos = infos[:self.max_return_files]

            return infos

    def process_full_files(self, include_active: bool = False) -> List[BulkFileInfo]:
        """
        Process full files: log and remove from tracking

        Args:
            include_active: Whether to include active (non-full) files

        Returns:
            List of processed file information objects
        """
        with self.lock:
            full_files = self.get_full_files(include_active)
            if not full_files:
                # self.logger.info("No full files to process")
                return []

            # Log to operations log
            for file_info in full_files:
                # Ensure writer is committed
                if file_info.file_name in self.file_writers:
                    try:
                        self.file_writers[file_info.file_name].commit()
                        file_info.batch_file = self.file_writers[file_info.file_name].batch_files
                    except Exception as e:
                        self.logger.error(f"Error committing writer: {e}")

                log_message = (
                    f"\n*****************\n"
                    f"FILE PROCESSED\n"
                    f"*****************\n"
                    f"File Name      : {file_info.file_name}\n"
                    f"File Path      : {file_info.file_path}\n"
                    f"Record Count   : {file_info.record_count}\n"
                    f"Min ID         : {file_info.min_id}\n"
                    f"Max ID         : {file_info.max_id}\n"
                    f"Created At     : {file_info.created_at}\n"
                    f"Last Updated   : {file_info.last_updated_at}\n"
                    f"batch_file : {file_info.batch_file}\n"
                    f"*****************"
                )

                self.operation_logger.info(log_message)

                # Remove from tracking
                self.file_info_table.pop(file_info.file_name, None)

                # Remove writer reference
                if file_info.file_name in self.file_writers:
                    del self.file_writers[file_info.file_name]

                # Remove lock
                if file_info.file_name in self.file_locks:
                    del self.file_locks[file_info.file_name]

            self._save_metadata()
            self.logger.info(f"Processed {len(full_files)} full files")
            return full_files

    def get_file_info(self, file_name: str = None) -> Union[BulkFileInfo, List[BulkFileInfo], None]:
        """
        Get file information

        Args:
            file_name: File name to get info for, or None for all files

        Returns:
            File information object or list of objects
        """
        with self.lock:
            if file_name is None:
                return list(self.file_info_table.values())
            return self.file_info_table.get(file_name)

    def get_stats(self) -> Dict:
        """Get statistics about file manager status"""
        with self.lock:
            total_records = sum(info.record_count for info in self.file_info_table.values())
            full_files = sum(1 for info in self.file_info_table.values() if info.is_full)
            active_files = sum(1 for info in self.file_info_table.values() if not info.is_full)
            active_writers = len(self.file_writers)

            return {
                "total_files": len(self.file_info_table),
                "full_files": full_files,
                "active_files": active_files,
                "active_writers": active_writers,
                "total_records": total_records,
                "max_files_limit": self.max_files,
                "can_create_new_file": active_files < self.max_files
            }

    def close(self):
        """Close all open writers and save metadata"""
        with self.lock:
            # Commit all writers
            for writer in self.file_writers.values():
                try:
                    writer.commit()
                except Exception as e:
                    self.logger.error(f"Error committing writer: {e}")

            # Clear writer references
            self.file_writers.clear()
            self.current_writer = None

            # Save metadata
            self._save_metadata()

        self.logger.info("MilvusBulkWriterManager has closed all files")

    def __del__(self):
        """Destructor to ensure all writers are properly closed"""
        try:
            self.close()
        except Exception:
            # Avoid raising exceptions in destructor
            pass
