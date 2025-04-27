import pymysql
from typing import Optional

class MySQLClient:
    def __init__(self, host: str, port: int, user: str, password: str, database: str):
        """
        初始化MySQL连接，只连一次
        """
        self.connection = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=True  # 自动提交，避免忘了commit
        )
        print("[INFO] MySQL connection established.")

    def get_id_by_filename(self, table_name: str, file_name: str) -> Optional[int]:
        """
        根据文件名查询对应的ID
        """
        try:
            with self.connection.cursor() as cursor:
                sql = f"SELECT id FROM `{table_name}` WHERE file_name = %s LIMIT 1"
                cursor.execute(sql, (file_name,))
                result = cursor.fetchone()
                if result:
                    return result['id']
                else:
                    return None
        except Exception as e:
            print(f"[ERROR] Query failed: {e}")
            return None

    def close(self):
        """
        关闭MySQL连接
        """
        if self.connection:
            self.connection.close()
            print("[INFO] MySQL connection closed.")
