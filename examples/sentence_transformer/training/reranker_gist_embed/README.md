# Training with RerankerCachedGISTEmbedLoss

This example demonstrates how to use `RerankerCachedGISTEmbedLoss` to train sentence transformer models using an external reranker API instead of a guide model.

## Overview

`RerankerCachedGISTEmbedLoss` combines the benefits of:
- **GIST-guided training**: Using a more accurate reranker to identify false negatives
- **Gradient caching**: Allowing for larger effective batch sizes with limited GPU memory
- **External reranker**: Leveraging powerful reranking models via API

## How it Works

### 1. Architecture

```
Training Model (SentenceTransformer)
           ↓
    Encode texts to embeddings
           ↓
    Compute similarities
           ↓
Reranker API (External Service)
           ↓
    Score query-document pairs
           ↓
    Filter false negatives
           ↓
    Calculate contrastive loss
```

### 2. Internal Process

1. **Text Extraction**: The loss function extracts raw texts from input tokens using the model's tokenizer
2. **Embedding Generation**: Creates embeddings using the training model (with gradient caching)
3. **Reranker Scoring**: Sends query-document pairs to the reranker API to get relevance scores
4. **False Negative Filtering**: Uses reranker scores to identify and mask false negatives
5. **Loss Calculation**: Computes contrastive loss with filtered negatives

### 3. Key Features

- **Tokenizer Handling**: Automatically handles tokenizer differences between training model and reranker
- **Batch Processing**: Efficiently batches API calls to the reranker
- **Async API Calls**: Uses async/await for concurrent API requests
- **Memory Efficiency**: Gradient caching allows training with large batch sizes

## Setup

### 1. Start the Reranker Server

First, ensure your reranker server is running:

```bash
python reranker_server.py
```

The server should be accessible at `http://localhost:8000` (or your configured URL).

### 2. Install Dependencies

```bash
pip install sentence-transformers aiohttp
```

### 3. Run Training

```bash
python train_with_reranker.py
```

## Configuration Options

### Loss Function Parameters

- `reranker_url`: URL of the reranker API server
- `mini_batch_size`: Size of mini-batches for gradient caching (affects memory usage)
- `reranker_batch_size`: Maximum batch size for each API call (splits larger batches automatically)
- `margin_strategy`: "absolute" or "relative" for false negative filtering
- `margin`: Threshold for filtering (0.0 = only filter exact matches)
- `temperature`: Temperature for scaling similarities
- `instruction`: Task instruction sent to the reranker
- `max_length`: Maximum token length for reranker
- `timeout`: API timeout in seconds

### API Usage

The loss function now uses the `/rerank_batch` endpoint for better efficiency:
- Multiple mini-batches are combined into a single API request (up to `reranker_batch_size`)
- If the batch API is not available (returns non-200 status), it falls back to individual calls
- This reduces network overhead and improves throughput

### Server Implementations

1. **Basic Server** (`reranker_server.py`): Original implementation
2. **Minimal Fix** (`reranker_server_minimal_fix.py`): Basic fixes with proper vLLM batching
3. **Improved Server** (`reranker_server_improved.py`): Adds caching, metrics, and batch processing
4. **Optimized Batch** (`reranker_server_optimized_batch.py`): Best vLLM batch utilization with:
   - `/rerank`: Standard single request endpoint
   - `/rerank_batch`: Intelligent grouping by instruction/max_length
   - `/rerank_batch_simple`: Maximum efficiency when all requests share same parameters

### Margin Strategies

1. **Absolute margin** (`margin_strategy="absolute"`):
   - Filters samples where: `reranker_score > (positive_score - margin)`
   - Example: If positive score is 0.9 and margin is 0.1, filters all with score > 0.8

2. **Relative margin** (`margin_strategy="relative"`):
   - Filters samples where: `reranker_score > (positive_score * (1 - margin))`
   - Example: If positive score is 0.9 and margin is 0.1, filters all with score > 0.81

## API Server Improvements

The current reranker server can be improved in several ways:

### 1. Performance Optimizations

```python
# Add request batching queue
from asyncio import Queue
import asyncio

class BatchProcessor:
    def __init__(self, batch_size=256, timeout=0.1):
        self.queue = Queue()
        self.batch_size = batch_size
        self.timeout = timeout
        
    async def process_batches(self):
        while True:
            batch = []
            deadline = asyncio.get_event_loop().time() + self.timeout
            
            while len(batch) < self.batch_size:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    break
                    
                try:
                    item = await asyncio.wait_for(
                        self.queue.get(), 
                        timeout=remaining
                    )
                    batch.append(item)
                except asyncio.TimeoutError:
                    break
                    
            if batch:
                # Process batch
                await self._process_batch(batch)
```

### 2. Caching Layer

```python
from functools import lru_cache
import hashlib

class RerankerCache:
    def __init__(self, max_size=10000):
        self.cache = {}
        self.max_size = max_size
        
    def get_key(self, query: str, doc: str, instruction: str) -> str:
        return hashlib.md5(
            f"{query}|{doc}|{instruction}".encode()
        ).hexdigest()
        
    def get(self, query: str, doc: str, instruction: str) -> float | None:
        key = self.get_key(query, doc, instruction)
        return self.cache.get(key)
        
    def set(self, query: str, doc: str, instruction: str, score: float):
        if len(self.cache) >= self.max_size:
            # Simple LRU: remove first item
            self.cache.pop(next(iter(self.cache)))
        key = self.get_key(query, doc, instruction)
        self.cache[key] = score
```

### 3. Enhanced Error Handling

```python
from fastapi import HTTPException
from pydantic import BaseModel

class ErrorResponse(BaseModel):
    error: str
    detail: str
    
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            detail=str(exc)
        ).dict()
    )
```

### 4. Monitoring and Metrics

```python
from prometheus_client import Counter, Histogram, generate_latest

# Metrics
request_count = Counter('reranker_requests_total', 'Total requests')
request_duration = Histogram('reranker_request_duration_seconds', 'Request duration')
batch_size_histogram = Histogram('reranker_batch_size', 'Batch sizes')

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

## Best Practices

1. **Batch Size Selection**:
   - Larger `mini_batch_size` = more memory efficient but slower
   - Larger `reranker_batch_size` = better API throughput but higher latency

2. **Margin Tuning**:
   - Start with `margin=0.0` (only filter exact matches)
   - Increase gradually if seeing too many false negatives
   - Monitor training metrics to find optimal value

3. **API Server Deployment**:
   - Use multiple workers: `uvicorn app:app --workers 4`
   - Consider load balancing for multiple GPU nodes
   - Implement request queuing for better throughput

4. **Training Tips**:
   - Use larger batch sizes (64-256) due to gradient caching
   - Monitor API response times and adjust timeouts
   - Consider implementing retry logic for failed API calls

## Troubleshooting

1. **API Timeout Errors**:
   - Increase `timeout` parameter
   - Reduce `reranker_batch_size`
   - Check server load and scaling

2. **Out of Memory**:
   - Reduce `mini_batch_size`
   - Reduce `per_device_train_batch_size`
   - Enable gradient checkpointing

3. **Slow Training**:
   - Increase `reranker_batch_size` (if API can handle it)
   - Use multiple API server instances
   - Enable API response caching

## Example Results

With proper configuration, RerankerCachedGISTEmbedLoss can achieve:
- Better negative sampling compared to standard MultipleNegativesRankingLoss
- Training with effective batch sizes of 1000+ on limited GPU memory
- Improved performance on retrieval benchmarks