from __future__ import annotations

"""
Optimized reranker server with proper vLLM batching for RerankerCachedGISTEmbedLoss
"""

import logging
import math
import gc
from typing import Dict, List, Optional, Tuple
from contextlib import asynccontextmanager
from collections import defaultdict
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
from vllm.inputs.data import TokensPrompt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and tokenizer
model = None
tokenizer = None
suffix_tokens = None
true_token = None
false_token = None
sampling_params = None


class RerankRequest(BaseModel):
    """Request model for reranking endpoint"""
    pairs: List[Tuple[str, str]] = Field(..., description="List of (query, document) pairs")
    instruction: str = Field(
        default="Given a web search query, retrieve relevant passages that answer the query",
        description="Task instruction for the reranker"
    )
    max_length: int = Field(default=8192, description="Maximum token length")


class RerankResponse(BaseModel):
    """Response model for reranking endpoint"""
    scores: List[float] = Field(..., description="Relevance scores for each pair")


class HealthResponse(BaseModel):
    """Response model for health check endpoint"""
    status: str = "healthy"
    model_loaded: bool = True


def format_instruction(instruction: str, query: str, doc: str) -> List[Dict[str, str]]:
    """Format the instruction for the model"""
    return [
        {"role": "system", "content": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."},
        {"role": "user", "content": f"<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {doc}"}
    ]


def process_inputs_batch(pairs: List[Tuple[str, str]], instruction: str, max_length: int) -> List[TokensPrompt]:
    """Process input pairs into model-ready format with proper batching"""
    messages = [format_instruction(instruction, query, doc) for query, doc in pairs]
    
    # Process each message individually to handle tokenization properly
    tokenized_messages = []
    for msg in messages:
        tokenized = tokenizer.apply_chat_template(
            [msg], 
            tokenize=True, 
            add_generation_prompt=False,
        )
        # Handle both list and single output formats
        if isinstance(tokenized[0], list):
            tokenized = tokenized[0]
        tokenized_messages.append(tokenized)
    
    # Truncate and add suffix
    processed_messages = []
    for tokens in tokenized_messages:
        truncated = tokens[:max_length - len(suffix_tokens)] + suffix_tokens
        processed_messages.append(TokensPrompt(prompt_token_ids=truncated))
    
    return processed_messages


def compute_logits_batch(messages: List[TokensPrompt]) -> List[float]:
    """Compute relevance scores from model outputs using vLLM batching"""
    # Process all messages in a single vLLM batch
    outputs = model.generate(messages, sampling_params, use_tqdm=False)
    scores = []
    
    for output in outputs:
        final_logits = output.outputs[0].logprobs[-1]
        
        # Get logits for true/false tokens
        true_logit = final_logits.get(true_token)
        false_logit = final_logits.get(false_token)
        
        # Handle missing logits with default values
        true_prob = math.exp(true_logit.logprob if true_logit else -10)
        false_prob = math.exp(false_logit.logprob if false_logit else -10)
        
        # Normalize to get relevance score
        score = true_prob / (true_prob + false_prob)
        scores.append(score)
    
    return scores


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage model lifecycle"""
    global model, tokenizer, suffix_tokens, true_token, false_token, sampling_params
    
    # Startup
    logger.info("Loading Qwen3-Reranker model...")
    
    try:
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Reranker-0.6B')
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
        if number_of_gpu == 0:
            logger.warning("No GPUs found, using CPU (will be slow)")
            model = LLM(
                model='Qwen/Qwen3-Reranker-0.6B',
                device='cpu',
                max_model_len=10000,
                gpu_memory_utilization=0.0,
            )
        else:
            model = LLM(
                model='Qwen/Qwen3-Reranker-0.6B',
                tensor_parallel_size=number_of_gpu,
                max_model_len=10000,
                enable_prefix_caching=True,
                gpu_memory_utilization=0.6
            )
        
        # Set sampling parameters
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=1,
            logprobs=20,
        )
        
        logger.info("Model loaded successfully!")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down model...")
    if model is not None:
        destroy_model_parallel()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Create FastAPI app
app = FastAPI(
    title="Qwen3 Reranker API - Optimized Batch",
    description="FastAPI server for Qwen3-Reranker model with optimized batching",
    version="2.0.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the server is healthy and model is loaded"""
    return HealthResponse(
        status="healthy",
        model_loaded=(model is not None)
    )


@app.post("/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest):
    """
    Rerank query-document pairs
    
    Args:
        request: RerankRequest containing pairs and optional instruction
        
    Returns:
        RerankResponse with relevance scores
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        logger.info(f"Reranking {len(request.pairs)} pairs")
        
        # Process inputs
        inputs = process_inputs_batch(
            request.pairs,
            request.instruction,
            request.max_length
        )
        
        # Compute scores with vLLM batching
        scores = compute_logits_batch(inputs)
        
        return RerankResponse(scores=scores)
        
    except Exception as e:
        logger.error(f"Error during reranking: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rerank_batch", response_model=List[RerankResponse])
async def rerank_batch(requests: List[RerankRequest]):
    """
    Batch reranking for multiple requests with intelligent grouping
    Groups requests by instruction and max_length for efficient vLLM batching
    
    Args:
        requests: List of RerankRequest objects
        
    Returns:
        List of RerankResponse objects
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Group requests by (instruction, max_length) for efficient batching
        grouped_requests = defaultdict(list)
        request_indices = defaultdict(list)
        
        for idx, request in enumerate(requests):
            key = (request.instruction, request.max_length)
            grouped_requests[key].extend(request.pairs)
            request_indices[key].append((idx, len(request.pairs)))
        
        logger.info(f"Processing {len(requests)} requests in {len(grouped_requests)} groups")
        
        # Process each group with same parameters
        all_results = {}
        
        for (instruction, max_length), pairs in grouped_requests.items():
            logger.info(f"Processing group with {len(pairs)} pairs")
            
            # Process all pairs in this group in one vLLM batch
            inputs = process_inputs_batch(pairs, instruction, max_length)
            scores = compute_logits_batch(inputs)
            
            # Store results for this group
            all_results[(instruction, max_length)] = scores
        
        # Reconstruct responses in original order
        responses = [None] * len(requests)
        
        for key, indices in request_indices.items():
            scores = all_results[key]
            score_idx = 0
            
            for request_idx, pair_count in indices:
                request_scores = scores[score_idx:score_idx + pair_count]
                responses[request_idx] = RerankResponse(scores=request_scores)
                score_idx += pair_count
        
        return responses
        
    except Exception as e:
        logger.error(f"Error during batch reranking: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rerank_batch_simple", response_model=List[RerankResponse])
async def rerank_batch_simple(requests: List[RerankRequest]):
    """
    Simple batch reranking - assumes all requests have same instruction and max_length
    Processes all pairs in a single vLLM batch for maximum efficiency
    
    Args:
        requests: List of RerankRequest objects (should have same instruction/max_length)
        
    Returns:
        List of RerankResponse objects
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not requests:
        return []
    
    try:
        # Validate all requests have same parameters
        first_instruction = requests[0].instruction
        first_max_length = requests[0].max_length
        
        for req in requests[1:]:
            if req.instruction != first_instruction or req.max_length != first_max_length:
                raise ValueError(
                    "All requests must have the same instruction and max_length for simple batch. "
                    "Use /rerank_batch for mixed parameters."
                )
        
        # Collect all pairs
        all_pairs = []
        request_lengths = []
        
        for request in requests:
            all_pairs.extend(request.pairs)
            request_lengths.append(len(request.pairs))
        
        logger.info(f"Processing {len(all_pairs)} pairs from {len(requests)} requests in single vLLM batch")
        
        # Process all pairs in one vLLM batch
        inputs = process_inputs_batch(all_pairs, first_instruction, first_max_length)
        all_scores = compute_logits_batch(inputs)
        
        # Split scores back into individual responses
        responses = []
        start_idx = 0
        
        for length in request_lengths:
            end_idx = start_idx + length
            scores = all_scores[start_idx:end_idx]
            responses.append(RerankResponse(scores=scores))
            start_idx = end_idx
        
        return responses
        
    except Exception as e:
        logger.error(f"Error during simple batch reranking: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)