from FlagEmbedding import BGEM3FlagModel
from typing import List, Tuple, Union, cast
import asyncio
from fastapi import FastAPI, Request, Response, HTTPException
from starlette.status import HTTP_504_GATEWAY_TIMEOUT
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from uuid import uuid4
import time
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import logging
import traceback

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

batch_size = 64  # gpu batch_size in order of your available vram
max_request = 10  # max request for future improvements on api calls / gpu batches (for now is pretty basic)
max_length = 10000  # max context length for embeddings and passages in re-ranker
max_q_length = 256  # max context lenght for questions in re-ranker
request_flush_timeout = .1  # flush time out for future improvements on api calls / gpu batches (for now is pretty basic)
rerank_weights = [0.4, 0.2, 0.4]  # re-rank score weights
request_time_out = 30  # Timeout threshold
gpu_time_out = 30  # gpu processing timeout threshold
port = 7300


class m3Wrapper:
    def __init__(self, model_name: str, device: str = 'cuda'):
        try:
            logger.info(f"Initializing model on device: {device}")
            self.model = BGEM3FlagModel(model_name, device=device, use_fp16=True if device != 'cpu' else False)
            logger.info("Model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            logger.error(traceback.format_exc())
            raise

    def embed(self, sentences: List[str]) -> Tuple[List[List[float]], List[dict]]:
        try:
            logger.info(f"Processing {len(sentences)} sentences for embedding")
            output = self.model.encode(sentences, return_dense=True, return_sparse=True, batch_size=batch_size,
                                       max_length=max_length)
            embeddings = output['dense_vecs']
            embeddings = embeddings.tolist()
            sparse_embeddings = []

            for weights in output['lexical_weights']:
                non_zero_indices = [int(i) for i in weights.keys()]
                non_zero_values = [float(v) for v in weights.values()]
                sparse_dict = dict(zip(non_zero_indices, non_zero_values))
                sparse_embeddings.append(sparse_dict)

            logger.info("Embedding completed successfully")
            return embeddings, sparse_embeddings
        except Exception as e:
            logger.error(f"Error in embed: {e}")
            logger.error(traceback.format_exc())
            raise

    def rerank(self, sentence_pairs: List[Tuple[str, str]]) -> List[float]:
        try:
            logger.info(f"Processing {len(sentence_pairs)} sentence pairs for reranking")
            scores = self.model.compute_score(
                sentence_pairs,
                batch_size=batch_size,
                max_query_length=max_q_length,
                max_passage_length=max_length,
                weights_for_different_modes=rerank_weights
            )['colbert+sparse+dense']
            logger.info("Reranking completed successfully")
            return scores
        except Exception as e:
            logger.error(f"Error in rerank: {e}")
            logger.error(traceback.format_exc())
            raise


class EmbedRequest(BaseModel):
    sentences: List[str]


class RerankRequest(BaseModel):
    sentence_pairs: List[Tuple[str, str]]


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    sparse_embeddings: List[dict]


class RerankResponse(BaseModel):
    scores: List[float]


class RequestProcessor:
    def __init__(self, model: m3Wrapper, max_request_to_flush: int, accumulation_timeout: float):
        self.model = model
        self.max_batch_size = max_request_to_flush
        self.accumulation_timeout = accumulation_timeout
        self.queue = asyncio.Queue()
        self.response_futures = {}
        self.processing_loop_task = None
        self.processing_loop_started = False  # Processing pool flag lazy init state
        self.executor = ThreadPoolExecutor(max_workers=2)  # 限制线程数
        self.gpu_lock = asyncio.Semaphore(1)  # Sem for gpu sync usage

    async def ensure_processing_loop_started(self):
        if not self.processing_loop_started:
            logger.info('starting processing_loop')
            self.processing_loop_task = asyncio.create_task(self.processing_loop())
            self.processing_loop_started = True

    async def processing_loop(self):
        while True:
            try:
                requests, request_types, request_ids = [], [], []
                start_time = asyncio.get_event_loop().time()

                while len(requests) < self.max_batch_size:
                    timeout = self.accumulation_timeout - (asyncio.get_event_loop().time() - start_time)
                    if timeout <= 0:
                        break

                    try:
                        req_data, req_type, req_id = await asyncio.wait_for(self.queue.get(), timeout=timeout)
                        requests.append(req_data)
                        request_types.append(req_type)
                        request_ids.append(req_id)
                    except asyncio.TimeoutError:
                        break

                if requests:
                    await self.process_requests_by_type(requests, request_types, request_ids)
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                logger.error(traceback.format_exc())

    async def process_requests_by_type(self, requests, request_types, request_ids):
        tasks = []
        for request_data, request_type, request_id in zip(requests, request_types, request_ids):
            try:
                if request_type == 'embed':
                    task = asyncio.create_task(
                        self.run_with_semaphore(self.model.embed, request_data.sentences, request_id))
                else:  # 'rerank'
                    task = asyncio.create_task(
                        self.run_with_semaphore(self.model.rerank, request_data.sentence_pairs, request_id))
                tasks.append(task)
            except Exception as e:
                logger.error(f"Error creating task for request {request_id}: {e}")
                if request_id in self.response_futures:
                    self.response_futures[request_id].set_exception(e)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def run_with_semaphore(self, func, data, request_id):
        try:
            async with self.gpu_lock:  # Wait for sem
                # 检查request_id是否还存在（防止客户端断开连接）
                if request_id not in self.response_futures:
                    logger.warning(f"Request {request_id} no longer exists, skipping")
                    return

                future = self.executor.submit(func, data)
                try:
                    result = await asyncio.wait_for(asyncio.wrap_future(future), timeout=gpu_time_out)
                    if request_id in self.response_futures and not self.response_futures[request_id].done():
                        self.response_futures[request_id].set_result(result)
                except asyncio.TimeoutError:
                    logger.error(f"GPU processing timeout for request {request_id}")
                    if request_id in self.response_futures and not self.response_futures[request_id].done():
                        self.response_futures[request_id].set_exception(TimeoutError("GPU processing timeout"))
                except Exception as e:
                    logger.error(f"GPU processing error for request {request_id}: {e}")
                    if request_id in self.response_futures and not self.response_futures[request_id].done():
                        self.response_futures[request_id].set_exception(e)
        except Exception as e:
            logger.error(f"Error in run_with_semaphore for request {request_id}: {e}")
            logger.error(traceback.format_exc())
            if request_id in self.response_futures and not self.response_futures[request_id].done():
                self.response_futures[request_id].set_exception(e)

    async def process_request(self, request_data: Union[EmbedRequest, RerankRequest], request_type: str):
        try:
            await self.ensure_processing_loop_started()
            request_id = str(uuid4())
            self.response_futures[request_id] = asyncio.Future()
            await self.queue.put((request_data, request_type, request_id))

            try:
                result = await self.response_futures[request_id]
                return result
            finally:
                # 清理响应futures
                if request_id in self.response_futures:
                    del self.response_futures[request_id]

        except Exception as e:
            logger.error(f"Error in process_request: {e}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


# 全局变量，延迟初始化
app = FastAPI()
model = None
processor = None


@app.on_event("startup")
async def startup_event():
    global model, processor
    try:
        logger.info("Starting up server...")
        # 尝试NPU，失败则降级到CPU
        try:
            model = m3Wrapper('/data/BAAI/bge-m3', device='npu:0')
        except Exception as npu_error:
            logger.warning(f"NPU initialization failed: {npu_error}")
            logger.info("Falling back to CPU...")
            model = m3Wrapper('/data/BAAI/bge-m3', device='cpu')

        processor = RequestProcessor(model, accumulation_timeout=request_flush_timeout,
                                     max_request_to_flush=max_request)
        logger.info("Server startup completed successfully")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        logger.error(traceback.format_exc())
        raise


# Adding a middleware returning a 504 error if the request processing time is above a certain threshold
@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    try:
        start_time = time.time()
        return await asyncio.wait_for(call_next(request), timeout=request_time_out)

    except asyncio.TimeoutError:
        process_time = time.time() - start_time
        logger.warning(f"Request timeout after {process_time} seconds")
        return JSONResponse({'detail': 'Request processing time exceeded limit',
                             'processing_time': process_time},
                            status_code=HTTP_504_GATEWAY_TIMEOUT)
    except Exception as e:
        logger.error(f"Middleware error: {e}")
        logger.error(traceback.format_exc())
        return JSONResponse({'detail': 'Internal server error in middleware'},
                            status_code=500)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/embeddings/", response_model=EmbedResponse)
async def get_embeddings(request: EmbedRequest):
    try:
        if not model or not processor:
            raise HTTPException(status_code=503, detail="Model not initialized yet")

        # 验证输入
        if not request.sentences or len(request.sentences) == 0:
            raise HTTPException(status_code=400, detail="No sentences provided")

        logger.info(f"Received embedding request for {len(request.sentences)} sentences")
        embeddings, sparse_embeddings = await processor.process_request(request, 'embed')
        return EmbedResponse(embeddings=embeddings, sparse_embeddings=sparse_embeddings)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_embeddings: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to process embeddings: {str(e)}")


@app.post("/rerank/", response_model=RerankResponse)
async def rerank(request: RerankRequest):
    try:
        if not model or not processor:
            raise HTTPException(status_code=503, detail="Model not initialized yet")

        # 验证输入
        if not request.sentence_pairs or len(request.sentence_pairs) == 0:
            raise HTTPException(status_code=400, detail="No sentence pairs provided")

        logger.info(f"Received rerank request for {len(request.sentence_pairs)} pairs")
        scores = await processor.process_request(request, 'rerank')
        return RerankResponse(scores=scores)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in rerank: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to process reranking: {str(e)}")


if __name__ == "__main__":
    # 多进程支持
    multiprocessing.freeze_support()

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=port)