import concurrent.futures
import time
from multiprocessing import Process, Value
from my_embedding_req import MyEmbeddingClient  # 替换为实际客户端模块


class ConcurrentTester:
    def __init__(self, host, port, concurrency=10, requests_per_thread=100):
        self.host = host
        self.port = port
        self.concurrency = concurrency
        self.requests_per_thread = requests_per_thread
        self.counter = Value('i', 0)  # 共享计数器
        self.latencies = []  # 存储延迟数据

    def worker(self, thread_id):
        """单个工作线程的执行函数"""
        text = ["我是一个配角一个小配" * 320] * 128
        local_count = 0

        for _ in range(self.requests_per_thread):
            start_time = time.perf_counter()

            try:
                client = MyEmbeddingClient(self.host, self.port)
                dense, sparse = client.embed_all(text)

                # 原子操作更新计数器
                with self.counter.get_lock():
                    self.counter.value += 1
                    current_count = self.counter.value

                end_time = time.perf_counter()
                latency = (end_time - start_time) * 1000  # 毫秒
                self.latencies.append(latency)

                if current_count % 10 == 0:
                    print(f"Thread-{thread_id}: Req {local_count + 1}/{self.requests_per_thread} | "
                          f"Total: {current_count} | Latency: {latency:.2f}ms")
            except Exception as e:
                print(f"Thread-{thread_id} failed: {str(e)}")

            local_count += 1

    def run(self):
        """启动并发测试"""
        print(f"Starting concurrent test with {self.concurrency} threads, "
              f"{self.concurrency * self.requests_per_thread} total requests")

        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            # 提交所有任务
            futures = [executor.submit(self.worker, i) for i in range(self.concurrency)]

            # 等待所有任务完成
            concurrent.futures.wait(futures)

        total_time = time.time() - start_time
        total_requests = self.concurrency * self.requests_per_thread

        # 计算性能指标
        avg_latency = sum(self.latencies) / len(self.latencies) if self.latencies else 0
        rps = total_requests / total_time if total_time > 0 else 0

        print("\n" + "=" * 50)
        print(f"Test completed in {total_time:.2f} seconds")
        print(f"Total requests: {total_requests}")
        print(f"Requests per second: {rps:.2f}")
        print(f"Average latency: {avg_latency:.2f}ms")
        print("=" * 50)


if __name__ == "__main__":
    # 配置参数
    HOST = "127.0.0.1"
    PORT = 7300
    CONCURRENCY = 5  # 并发线程数
    REQUESTS_PER_THREAD = 20  # 每个线程请求数

    tester = ConcurrentTester(
        host=HOST,
        port=PORT,
        concurrency=CONCURRENCY,
        requests_per_thread=REQUESTS_PER_THREAD
    )

    tester.run()