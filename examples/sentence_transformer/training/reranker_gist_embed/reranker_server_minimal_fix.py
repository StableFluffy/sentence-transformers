from __future__ import annotations

"""
Minimal fixes for the original reranker_server.py to work better with RerankerCachedGISTEmbedLoss
"""

import logging
import math
import gc
from typing import Dict, List, Optional, Tuple
from contextlib import asynccontextmanager
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


def process_inputs(pairs: List[Tuple[str, str]], instruction: str, max_length: int) -> List[TokensPrompt]:
    """Process input pairs into model-ready format"""
    messages = [format_instruction(instruction, query, doc) for query, doc in pairs]
    
    # FIX: Handle tokenizer properly - it returns list of lists for batch
    tokenized = tokenizer.apply_chat_template(
        messages, 
        tokenize=True, 
        add_generation_prompt=False,
        padding=True,  # Add padding for batch processing
        return_tensors=None,  # Return lists
    )
    
    # If tokenizer returns a single list, split it appropriately
    if isinstance(tokenized[0], int):
        # This means we got a flat list, need to handle differently
        # For now, process one by one
        tokenized_messages = []
        for msg in messages:
            single_tokenized = tokenizer.apply_chat_template(
                [msg], 
                tokenize=True, 
                add_generation_prompt=False,
            )
            tokenized_messages.append(single_tokenized)
        tokenized = tokenized_messages
    
    # Truncate and add suffix
    messages = [ele[:max_length - len(suffix_tokens)] + suffix_tokens for ele in tokenized]
    messages = [TokensPrompt(prompt_token_ids=ele) for ele in messages]
    return messages


def compute_logits(messages: List[TokensPrompt]) -> List[float]:
    """Compute relevance scores from model outputs"""
    outputs = model.generate(messages, sampling_params, use_tqdm=False)
    scores = []
    
    for i in range(len(outputs)):
        final_logits = outputs[i].outputs[0].logprobs[-1]
        
        # Get logits for true/false tokens
        true_logit = final_logits.get(true_token, None)
        false_logit = final_logits.get(false_token, None)
        
        # Handle missing logits
        if true_logit is None:
            true_logit = type('', (), {'logprob': -10})
        else:
            true_logit = true_logit
            
        if false_logit is None:
            false_logit = type('', (), {'logprob': -10})
        else:
            false_logit = false_logit
        
        # Convert to probabilities
        true_score = math.exp(true_logit.logprob)
        false_score = math.exp(false_logit.logprob)
        
        # Normalize to get relevance score
        score = true_score / (true_score + false_score)
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
            # For CPU, we need different settings
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
            # FIX: Use logits_processors instead of allowed_token_ids for compatibility
            logits_processors=None,  # We'll filter in post-processing
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
    title="Qwen3 Reranker API",
    description="FastAPI server for Qwen3-Reranker model",
    version="1.0.1",
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
        # Log request info
        logger.info(f"Reranking {len(request.pairs)} pairs with instruction: {request.instruction[:50]}...")
        
        # Process inputs
        inputs = process_inputs(
            request.pairs,
            request.instruction,
            request.max_length
        )
        
        # Compute scores
        scores = compute_logits(inputs)
        
        logger.info(f"Reranking completed. Scores: {scores[:5]}..." if len(scores) > 5 else f"Scores: {scores}")
        
        return RerankResponse(scores=scores)
        
    except Exception as e:
        logger.error(f"Error during reranking: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rerank_batch", response_model=List[RerankResponse])
async def rerank_batch(requests: List[RerankRequest]):
    """
    Batch reranking for multiple requests
    
    Args:
        requests: List of RerankRequest objects
        
    Returns:
        List of RerankResponse objects
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        responses = []
        for request in requests:
            inputs = process_inputs(
                request.pairs,
                request.instruction,
                request.max_length
            )
            scores = compute_logits(inputs)
            responses.append(RerankResponse(scores=scores))
        
        return responses
        
    except Exception as e:
        logger.error(f"Error during batch reranking: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)