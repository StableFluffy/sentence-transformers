from __future__ import annotations

import asyncio
import gc
import hashlib
import logging
import math
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Tuple

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
from vllm.inputs.data import TokensPrompt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
request_count = Counter("reranker_requests_total", "Total requests", ["endpoint"])
request_duration = Histogram("reranker_request_duration_seconds", "Request duration", ["endpoint"])
batch_size_histogram = Histogram("reranker_batch_size", "Batch sizes")
cache_hits = Counter("reranker_cache_hits_total", "Cache hits")
cache_misses = Counter("reranker_cache_misses_total", "Cache misses")

# Global variables
model = None
tokenizer = None
suffix_tokens = None
true_token = None
false_token = None
sampling_params = None
cache = None
batch_processor = None


class RerankRequest(BaseModel):
    """Request model for reranking endpoint"""

    pairs: List[Tuple[str, str]] = Field(..., description="List of (query, document) pairs")
    instruction: str = Field(
        default="Given a web search query, retrieve relevant passages that answer the query",
        description="Task instruction for the reranker",
    )
    max_length: int = Field(default=8192, description="Maximum token length")


class RerankResponse(BaseModel):
    """Response model for reranking endpoint"""

    scores: List[float] = Field(..., description="Relevance scores for each pair")


class HealthResponse(BaseModel):
    """Response model for health check endpoint"""

    status: str = "healthy"
    model_loaded: bool = True
    cache_size: int = 0
    cache_hit_rate: float = 0.0


class ErrorResponse(BaseModel):
    """Response model for errors"""

    error: str
    detail: str


class LRUCache:
    """Simple LRU cache implementation"""

    def __init__(self, max_size: int = 10000):
        self.cache: OrderedDict[str, float] = OrderedDict()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def get_key(self, query: str, doc: str, instruction: str) -> str:
        """Generate cache key"""
        return hashlib.md5(f"{instruction}|{query}|{doc}".encode()).hexdigest()

    def get(self, query: str, doc: str, instruction: str) -> Optional[float]:
        """Get value from cache"""
        key = self.get_key(query, doc, instruction)
        if key in self.cache:
            self.hits += 1
            cache_hits.inc()
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        self.misses += 1
        cache_misses.inc()
        return None

    def set(self, query: str, doc: str, instruction: str, score: float):
        """Set value in cache"""
        key = self.get_key(query, doc, instruction)
        self.cache[key] = score
        self.cache.move_to_end(key)
        # Remove oldest if over capacity
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def get_hit_rate(self) -> float:
        """Get cache hit rate"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class BatchProcessor:
    """Batch processor for efficient request handling"""

    def __init__(self, batch_size: int = 256, timeout: float = 0.1):
        self.batch_size = batch_size
        self.timeout = timeout
        self.queue: asyncio.Queue = asyncio.Queue()
        self.processing_task = None

    async def start(self):
        """Start the batch processor"""
        self.processing_task = asyncio.create_task(self._process_loop())

    async def stop(self):
        """Stop the batch processor"""
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass

    async def _process_loop(self):
        """Main processing loop"""
        while True:
            batch = []
            futures = []
            deadline = asyncio.get_event_loop().time() + self.timeout

            # Collect items for batch
            while len(batch) < self.batch_size:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    break

                try:
                    item, future = await asyncio.wait_for(self.queue.get(), timeout=remaining)
                    batch.append(item)
                    futures.append(future)
                except asyncio.TimeoutError:
                    break

            if batch:
                batch_size_histogram.observe(len(batch))
                try:
                    # Process batch
                    results = await self._process_batch(batch)
                    # Send results
                    for result, future in zip(results, futures):
                        future.set_result(result)
                except Exception as e:
                    # Send error to all futures
                    for future in futures:
                        future.set_exception(e)

    async def _process_batch(self, batch: List[dict]) -> List[float]:
        """Process a batch of reranking requests"""
        # Extract unique instruction (assuming all in batch have same instruction)
        instruction = batch[0]["instruction"]
        max_length = batch[0]["max_length"]

        # Check cache first
        cached_results = []
        uncached_items = []
        uncached_indices = []

        for i, item in enumerate(batch):
            cached_score = cache.get(item["query"], item["doc"], instruction)
            if cached_score is not None:
                cached_results.append((i, cached_score))
            else:
                uncached_items.append(item)
                uncached_indices.append(i)

        # Process uncached items
        if uncached_items:
            messages = format_batch_messages(uncached_items, instruction, max_length)
            scores = compute_batch_logits(messages)

            # Cache results
            for item, score in zip(uncached_items, scores):
                cache.set(item["query"], item["doc"], instruction, score)

        # Combine results
        results = [0.0] * len(batch)
        for i, score in cached_results:
            results[i] = score
        for i, score in zip(uncached_indices, scores if uncached_items else []):
            results[i] = score

        return results

    async def submit(self, query: str, doc: str, instruction: str, max_length: int) -> float:
        """Submit a single item for processing"""
        future = asyncio.get_event_loop().create_future()
        item = {"query": query, "doc": doc, "instruction": instruction, "max_length": max_length}
        await self.queue.put((item, future))
        return await future


def format_instruction(instruction: str, query: str, doc: str) -> List[Dict[str, str]]:
    """Format the instruction for the model"""
    return [
        {
            "role": "system",
            "content": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".",
        },
        {"role": "user", "content": f"<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {doc}"},
    ]


def format_batch_messages(items: List[dict], instruction: str, max_length: int) -> List[TokensPrompt]:
    """Format batch of messages for model"""
    messages = [format_instruction(instruction, item["query"], item["doc"]) for item in items]
    messages = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, enable_thinking=False
    )
    messages = [ele[: max_length - len(suffix_tokens)] + suffix_tokens for ele in messages]
    messages = [TokensPrompt(prompt_token_ids=ele) for ele in messages]
    return messages


def compute_batch_logits(messages: List[TokensPrompt]) -> List[float]:
    """Compute relevance scores from model outputs"""
    outputs = model.generate(messages, sampling_params, use_tqdm=False)
    scores = []

    for i in range(len(outputs)):
        final_logits = outputs[i].outputs[0].logprobs[-1]

        # Get logits for true/false tokens
        true_logit = final_logits.get(true_token, type("", (), {"logprob": -10})).logprob
        false_logit = final_logits.get(false_token, type("", (), {"logprob": -10})).logprob

        # Convert to probabilities
        true_score = math.exp(true_logit)
        false_score = math.exp(false_logit)

        # Normalize to get relevance score
        score = true_score / (true_score + false_score)
        scores.append(score)

    return scores


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage model lifecycle"""
    global model, tokenizer, suffix_tokens, true_token, false_token, sampling_params, cache, batch_processor

    # Startup
    logger.info("Loading Qwen3-Reranker model...")

    try:
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-0.6B")
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token

        # Prepare suffix tokens
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

        # Get true/false token IDs
        true_token = tokenizer("yes", add_special_tokens=False).input_ids[0]
        false_token = tokenizer("no", add_special_tokens=False).input_ids[0]

        # Initialize model
        number_of_gpu = torch.cuda.device_count()
        model = LLM(
            model="Qwen/Qwen3-Reranker-0.6B",
            tensor_parallel_size=number_of_gpu,
            max_model_len=10000,
            enable_prefix_caching=True,
            gpu_memory_utilization=0.6,
        )

        # Set sampling parameters
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=1,
            logprobs=20,
            allowed_token_ids=[true_token, false_token],
        )

        # Initialize cache
        cache = LRUCache(max_size=10000)

        # Initialize batch processor
        batch_processor = BatchProcessor(batch_size=256, timeout=0.1)
        await batch_processor.start()

        logger.info("Model loaded successfully!")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down model...")
    if batch_processor:
        await batch_processor.stop()
    destroy_model_parallel()
    gc.collect()
    torch.cuda.empty_cache()


# Create FastAPI app
app = FastAPI(
    title="Qwen3 Reranker API",
    description="High-performance FastAPI server for Qwen3-Reranker model",
    version="2.0.0",
    lifespan=lifespan,
)


# Exception handlers
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(error="Internal Server Error", detail=str(exc)).dict(),
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the server is healthy and model is loaded"""
    return HealthResponse(
        status="healthy",
        model_loaded=(model is not None),
        cache_size=len(cache.cache) if cache else 0,
        cache_hit_rate=cache.get_hit_rate() if cache else 0.0,
    )


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()


@app.post("/rerank", response_model=RerankResponse)
@request_duration.labels(endpoint="rerank").time()
async def rerank(request: RerankRequest):
    """
    Rerank query-document pairs

    Args:
        request: RerankRequest containing pairs and optional instruction

    Returns:
        RerankResponse with relevance scores
    """
    request_count.labels(endpoint="rerank").inc()

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        start_time = time.time()

        # Process pairs using batch processor
        tasks = []
        for query, doc in request.pairs:
            task = batch_processor.submit(query, doc, request.instruction, request.max_length)
            tasks.append(task)

        # Wait for all results
        scores = await asyncio.gather(*tasks)

        logger.info(f"Processed {len(request.pairs)} pairs in {time.time() - start_time:.2f}s")

        return RerankResponse(scores=scores)

    except Exception as e:
        logger.error(f"Error during reranking: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rerank_sync", response_model=RerankResponse)
@request_duration.labels(endpoint="rerank_sync").time()
async def rerank_sync(request: RerankRequest):
    """
    Synchronous reranking for compatibility (processes immediately without batching)

    Args:
        request: RerankRequest containing pairs and optional instruction

    Returns:
        RerankResponse with relevance scores
    """
    request_count.labels(endpoint="rerank_sync").inc()

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Check cache first
        scores = []
        uncached_pairs = []
        uncached_indices = []

        for i, (query, doc) in enumerate(request.pairs):
            cached_score = cache.get(query, doc, request.instruction)
            if cached_score is not None:
                scores.append(cached_score)
            else:
                scores.append(None)
                uncached_pairs.append({"query": query, "doc": doc})
                uncached_indices.append(i)

        # Process uncached pairs
        if uncached_pairs:
            messages = format_batch_messages(uncached_pairs, request.instruction, request.max_length)
            uncached_scores = compute_batch_logits(messages)

            # Update scores and cache
            for idx, score in zip(uncached_indices, uncached_scores):
                scores[idx] = score
                cache.set(
                    uncached_pairs[uncached_indices.index(idx)]["query"],
                    uncached_pairs[uncached_indices.index(idx)]["doc"],
                    request.instruction,
                    score,
                )

        return RerankResponse(scores=scores)

    except Exception as e:
        logger.error(f"Error during synchronous reranking: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear_cache")
async def clear_cache():
    """Clear the reranker cache"""
    if cache:
        cache.cache.clear()
        cache.hits = 0
        cache.misses = 0
        return {"message": "Cache cleared successfully"}
    else:
        raise HTTPException(status_code=503, detail="Cache not initialized")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)