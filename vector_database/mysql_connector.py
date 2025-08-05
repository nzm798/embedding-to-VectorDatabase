import pymysql
from dbutils.pooled_db import PooledDB
from typing import Optional


class MySQLClient:
    def __init__(self, host: str, port: int, user: str, password: str, database: str, table_name: str):
        """
        初始化MySQL连接，只连一次
        """
        self.pool = PooledDB(
            creator=pymysql,  # 使用pymysql作为连接器
            mincached=8,  # 初始化时，链接池中至少创建的链接，0表示不创建
            maxcached=16,  # 连接池允许的最大连接数，0和None表示不限制连接数
            maxshared=16,
            maxconnections=32,
            blocking=True,  # 连接池中如果没有可用连接后，是否阻塞等待。True，等待；False，不等待然后报错
            host=host,
            port=port,
            user=user,
            password=password,
            db=database,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=True
        )

        self.table_name = table_name
        print("[INFO] MySQL connection pool established.")

    def get_conn(self):
        try:
            conn = self.pool.connection()
            cursor = conn.cursor()
            return conn, cursor
        except Exception as e:
            print(f"[ERROR] Failed to get DB connection: {e}")
            return None, None

    def get_id_by_filename(self, file_name: str, conn, cursor) -> Optional[int]:
        """
        根据文件名查询对应的ID
        """
        if not conn or not cursor:
            print(f"[ERROR] Invalid database connection or cursor.")
            return None
        try:
            sql = f"SELECT id FROM `{self.table_name}` WHERE name = %s LIMIT 1"
            cursor.execute(sql, (file_name,))
            result = cursor.fetchone()
            if result:
                return result['id']
            else:
                return None
        except Exception as e:
            print(f"[ERROR] Query failed: {e}")
            return None
    def get_files_by_knowledge(self, knowledge_base_id: int, conn, cursor) -> Optional[list]:
        """
        查询在对应知识库ID下的所有数据
        """
        if not conn or not cursor:
            print(f"[ERROR] Invalid database connection or cursor.")
            return None
        try:
            sql = f"SELECT * FROM `{self.table_name}` WHERE konwledge_base_id = %s ORDER BY id ASC"
            cursor.execute(sql, (knowledge_base_id,))
            results = cursor.fetchall()
            if results:
                print(f"[INFO] Get knowledge base ID: {knowledge_base_id} ,number of results: {len(results)}\n")
                return results
            else:
                return None
        except Exception as e:
            print(f"[ERROR] Query failed: {e}")
            return None

    def close(self):
        """
        关闭MySQL连接
        """
        if self.pool:
            self.pool.close()
            print("[INFO] MySQL connection pool closed.")


if __name__=="__main__":
    mysql_client = MySQLClient(
        host="192.168.100.9",
        port=3306,
        user="szzf",
        password="jRzZHvnjRm1kJ9fRj5SL",
        database="dimension_beijing_xicheng",
        table_name="knowledge_document_library"
    )
    conn, cursor = mysql_client.get_conn()
    # print(mysql_client.get_id_by_filename("合肥市延长集中供暖_37765.pdf"))
    mysql_client.get_files_by_knowledge(123,conn,cursor)
