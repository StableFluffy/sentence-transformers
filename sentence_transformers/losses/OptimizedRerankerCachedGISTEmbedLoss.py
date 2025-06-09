from __future__ import annotations

import asyncio
from collections.abc import Iterable, Iterator
from contextlib import nullcontext
from functools import partial
from typing import Any, Literal

import aiohttp
import torch
import tqdm
from torch import Tensor, nn
from torch.utils.checkpoint import get_device_states, set_device_states

from sentence_transformers import SentenceTransformer
from sentence_transformers.models import StaticEmbedding


class RandContext:
    """
    Random-state context manager class. Reference: https://github.com/luyug/GradCache.

    This class will back up the pytorch's random state during initialization. Then when the context is activated,
    the class will set up the random state with the backed-up one.
    """

    def __init__(self, *tensors) -> None:
        self.fwd_cpu_state = torch.get_rng_state()
        self.fwd_gpu_devices, self.fwd_gpu_states = get_device_states(*tensors)

    def __enter__(self) -> None:
        self._fork = torch.random.fork_rng(devices=self.fwd_gpu_devices, enabled=True)
        self._fork.__enter__()
        torch.set_rng_state(self.fwd_cpu_state)
        set_device_states(self.fwd_gpu_devices, self.fwd_gpu_states)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._fork.__exit__(exc_type, exc_val, exc_tb)
        self._fork = None


def _backward_hook(
    grad_output: Tensor,
    sentence_features: Iterable[dict[str, Tensor]],
    loss_obj: OptimizedRerankerCachedGISTEmbedLoss,
) -> None:
    """A backward hook to backpropagate the cached gradients mini-batch by mini-batch."""
    assert loss_obj.cache is not None
    assert loss_obj.random_states is not None
    with torch.enable_grad():
        for sentence_feature, grad, random_states in zip(sentence_features, loss_obj.cache, loss_obj.random_states):
            for (reps_mb, _), grad_mb in zip(
                loss_obj.embed_minibatch_iter(
                    sentence_feature=sentence_feature,
                    with_grad=True,
                    copy_random_state=False,
                    random_states=random_states,
                ),
                grad,
            ):
                surrogate = torch.dot(reps_mb.flatten(), grad_mb.flatten()) * grad_output
                surrogate.backward()


class OptimizedRerankerCachedGISTEmbedLoss(nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        reranker_url: str,
        temperature: float = 0.01,
        mini_batch_size: int = 32,
        show_progress_bar: bool = False,
        margin_strategy: Literal["absolute", "relative"] = "absolute",
        margin: float = 0.0,
        reranker_batch_size: int = 128,
        instruction: str = "Given a web search query, retrieve relevant passages that answer the query",
        max_length: int = 8192,
        timeout: int = 30,
        use_simple_batch: bool = True,
        # Optimization parameters
        similarity_threshold: float = 0.3,
        top_k_ratio: float = 0.1,
        min_top_k: int = 10,
        max_candidates: int = 100,
        hybrid_threshold: float = 0.5,
    ) -> None:
        """
        Optimized version of RerankerCachedGISTEmbedLoss that reduces reranker API calls.

        Key optimizations:
        1. Pre-filters candidates using embedding similarity
        2. Only reranks top-k candidates likely to be hard negatives
        3. Uses hybrid approach: embeddings for easy cases, reranker for hard cases

        Args:
            model: SentenceTransformer model to train
            reranker_url: URL of the reranker API server
            temperature: Temperature parameter to scale the cosine similarities
            mini_batch_size: Mini-batch size for the forward pass
            show_progress_bar: If True, show progress bar during training
            margin_strategy: Strategy for false negative filtering ("absolute" or "relative")
            margin: Margin value for filtering negatives
            reranker_batch_size: Maximum batch size for each reranker API call
            instruction: Task instruction for the reranker
            max_length: Maximum token length for reranker
            timeout: Timeout for API calls in seconds
            use_simple_batch: Use optimized batch endpoint for better vLLM efficiency
            similarity_threshold: Minimum similarity to consider for reranking (filters obvious negatives)
            top_k_ratio: Ratio of candidates to rerank (e.g., 0.1 = top 10%)
            min_top_k: Minimum number of candidates to rerank per query
            max_candidates: Maximum candidates to consider for reranking
            hybrid_threshold: Similarity threshold to decide between embedding vs reranker
        """
        super().__init__()
        if isinstance(model[0], StaticEmbedding):
            raise ValueError(
                "OptimizedRerankerCachedGISTEmbedLoss is not compatible with StaticEmbedding models."
            )
        self.model = model
        self.reranker_url = reranker_url
        self.temperature = temperature
        self.similarity_fct = nn.CosineSimilarity(dim=-1)
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.mini_batch_size = mini_batch_size
        self.cache: list[list[Tensor]] | None = None
        self.random_states: list[list[RandContext]] | None = None
        self.show_progress_bar = show_progress_bar
        self.tokenizer = model.tokenizer
        if margin_strategy not in ("absolute", "relative"):
            raise ValueError("margin_strategy must be 'absolute' or 'relative'.")
        self.margin_strategy = margin_strategy
        self.margin = margin
        self.reranker_batch_size = reranker_batch_size
        self.instruction = instruction
        self.max_length = max_length
        self.timeout = timeout
        self.use_simple_batch = use_simple_batch
        # Optimization parameters
        self.similarity_threshold = similarity_threshold
        self.top_k_ratio = top_k_ratio
        self.min_top_k = min_top_k
        self.max_candidates = max_candidates
        self.hybrid_threshold = hybrid_threshold

    def sim_matrix(self, embed1: Tensor, embed2: Tensor) -> Tensor:
        return self.similarity_fct(embed1.unsqueeze(1), embed2.unsqueeze(0))

    async def _call_reranker_async(self, session: aiohttp.ClientSession, pairs: list[tuple[str, str]]) -> list[float]:
        """Call reranker API asynchronously."""
        try:
            async with session.post(
                f"{self.reranker_url}/rerank",
                json={
                    "pairs": pairs,
                    "instruction": self.instruction,
                    "max_length": self.max_length,
                },
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data["scores"]
                else:
                    raise RuntimeError(f"Reranker API error: {response.status}")
        except Exception as e:
            raise RuntimeError(f"Failed to call reranker API: {e}")

    def select_candidates_for_reranking(
        self,
        sim_scores: Tensor,
        positive_indices: Tensor,
        query_idx: int = None
    ) -> tuple[Tensor, Tensor]:
        """
        Select candidates for reranking based on embedding similarity.
        Returns indices and a mask indicating which scores need reranking.
        """
        batch_size = sim_scores.shape[0] if len(sim_scores.shape) > 1 else 1

        # Initialize mask (True = needs reranking)
        needs_reranking = torch.zeros_like(sim_scores, dtype=torch.bool)

        if len(sim_scores.shape) == 2:  # Matrix case
            for i in range(batch_size):
                # Get positive score for this query
                pos_idx = positive_indices[i] if positive_indices is not None else i
                positive_score = sim_scores[i, pos_idx]

                # Step 1: Filter by similarity threshold
                candidates = sim_scores[i] > self.similarity_threshold

                # Step 2: Filter by hybrid threshold (close to positive)
                close_to_positive = sim_scores[i] > (positive_score * self.hybrid_threshold)
                candidates = candidates & close_to_positive

                # Step 3: Select top-k from remaining candidates
                candidate_scores = sim_scores[i].clone()
                candidate_scores[~candidates] = -float('inf')

                k = max(self.min_top_k, int(candidates.sum() * self.top_k_ratio))
                k = min(k, self.max_candidates, candidates.sum())

                if k > 0:
                    _, top_indices = torch.topk(candidate_scores, k)
                    needs_reranking[i, top_indices] = True

                # Always include positive sample
                needs_reranking[i, pos_idx] = True
        else:  # Vector case
            # Similar logic for 1D case
            pos_idx = positive_indices if positive_indices is not None else query_idx
            positive_score = sim_scores[pos_idx] if pos_idx is not None else sim_scores.max()

            candidates = (sim_scores > self.similarity_threshold) & \
                        (sim_scores > (positive_score * self.hybrid_threshold))

            k = max(self.min_top_k, int(candidates.sum() * self.top_k_ratio))
            k = min(k, self.max_candidates, candidates.sum())

            if k > 0:
                candidate_scores = sim_scores.clone()
                candidate_scores[~candidates] = -float('inf')
                _, top_indices = torch.topk(candidate_scores, k)
                needs_reranking[top_indices] = True

        return needs_reranking

    def call_reranker_selective(
        self,
        queries: list[str],
        documents: list[str],
        sim_matrix_tensor: Tensor,
        positive_indices: Tensor = None
    ) -> Tensor:
        """
        Selectively call reranker only for hard negatives identified by embedding similarity.
        Returns a matrix combining embedding scores and reranker scores.
        """
        reranker_scores = sim_matrix_tensor.clone()  # Start with embedding scores

        # Select candidates for each query
        needs_reranking = self.select_candidates_for_reranking(
            sim_matrix_tensor,
            positive_indices
        )

        # Collect pairs that need reranking
        pairs_to_rerank = []
        pair_indices = []

        for i in range(len(queries)):
            for j in range(len(documents)):
                if needs_reranking[i, j]:
                    pairs_to_rerank.append((queries[i], documents[j]))
                    pair_indices.append((i, j))

        if not pairs_to_rerank:
            return reranker_scores  # No reranking needed

        # Batch reranker calls
        async def process_reranking():
            async with aiohttp.ClientSession() as session:
                # Process in batches
                all_scores = []
                for i in range(0, len(pairs_to_rerank), self.reranker_batch_size):
                    batch = pairs_to_rerank[i:i + self.reranker_batch_size]
                    if self.use_simple_batch and len(batch) > 1:
                        # Use batch endpoint
                        batch_request = [{
                            "pairs": batch,
                            "instruction": self.instruction,
                            "max_length": self.max_length,
                        }]
                        async with session.post(
                            f"{self.reranker_url}/rerank_batch_simple",
                            json=batch_request,
                            timeout=aiohttp.ClientTimeout(total=self.timeout),
                        ) as response:
                            if response.status == 200:
                                data = await response.json()
                                all_scores.extend(data[0]["scores"])
                            else:
                                # Fallback to individual calls
                                scores = await self._call_reranker_async(session, batch)
                                all_scores.extend(scores)
                    else:
                        scores = await self._call_reranker_async(session, batch)
                        all_scores.extend(scores)
                return all_scores

        # Get reranker scores
        reranked_scores = asyncio.run(process_reranking())

        # Update only the reranked positions
        for (i, j), score in zip(pair_indices, reranked_scores):
            reranker_scores[i, j] = score

        return reranker_scores

    def embed_minibatch(
        self,
        sentence_feature: dict[str, Tensor],
        begin: int,
        end: int,
        with_grad: bool,
        copy_random_state: bool,
        random_state: RandContext | None = None,
    ) -> tuple[Tensor, RandContext | None]:
        """Do forward pass on a minibatch of the input features and return corresponding embeddings."""
        grad_context = nullcontext if with_grad else torch.no_grad
        random_state_context = nullcontext() if random_state is None else random_state
        sentence_feature_minibatch = {k: v[begin:end] for k, v in sentence_feature.items()}
        with random_state_context:
            with grad_context():
                random_state = RandContext(*sentence_feature_minibatch.values()) if copy_random_state else None
                reps = self.model(sentence_feature_minibatch)["sentence_embedding"]  # (mbsz, hdim)

        return reps, random_state

    def embed_minibatch_iter(
        self,
        sentence_feature: dict[str, Tensor],
        with_grad: bool,
        copy_random_state: bool,
        random_states: list[RandContext] | None = None,
    ) -> Iterator[tuple[Tensor, RandContext | None]]:
        """Do forward pass on all the minibatches of the input features and yield corresponding embeddings."""
        input_ids: Tensor = sentence_feature["input_ids"]
        bsz, _ = input_ids.shape
        for i, b in enumerate(
            tqdm.trange(
                0,
                bsz,
                self.mini_batch_size,
                desc="Embed mini-batches",
                disable=not self.show_progress_bar,
            )
        ):
            e = b + self.mini_batch_size
            reps, random_state = self.embed_minibatch(
                sentence_feature=sentence_feature,
                begin=b,
                end=e,
                with_grad=with_grad,
                copy_random_state=copy_random_state,
                random_state=None if random_states is None else random_states[i],
            )
            yield reps, random_state  # reps: (mbsz, hdim)

    def calculate_loss_and_cache_gradients(self, reps: list[list[Tensor]], all_texts: list[list[str]]) -> Tensor:
        """Calculate the cross-entropy loss and cache the gradients wrt. the embeddings."""
        loss = self.calculate_loss(reps, all_texts, with_backward=True)
        loss = loss.detach().requires_grad_()

        self.cache = [[r.grad for r in rs] for rs in reps]

        return loss

    def calculate_loss(self, reps: list[list[Tensor]], all_texts: list[list[str]], with_backward: bool = False) -> Tensor:
        """Calculate the cross-entropy loss using selective reranking."""
        # Concatenate embeddings and texts along the batch dimension
        concatenated_reps = [torch.cat(rep, dim=0) for rep in reps]
        concatenated_texts = [sum(texts, []) for texts in all_texts]  # Flatten text lists

        labels = torch.arange(concatenated_reps[0].size(0)).long().to(concatenated_reps[0].device)
        batch_size = concatenated_reps[0].shape[0]

        losses: list[torch.Tensor] = []
        for b in tqdm.trange(
            0,
            batch_size,
            self.mini_batch_size,
            desc="Calculating loss",
            disable=not self.show_progress_bar,
        ):
            e = b + self.mini_batch_size

            # Get texts for current mini-batch
            anchor_texts = concatenated_texts[0][b:e]
            positive_texts = concatenated_texts[1]
            all_anchor_texts = concatenated_texts[0]

            # First compute embedding similarities
            ap_sim = self.sim_matrix(concatenated_reps[0][b:e], concatenated_reps[1])
            aa_sim = self.sim_matrix(concatenated_reps[0][b:e], concatenated_reps[0])
            pp_sim = self.sim_matrix(concatenated_reps[1][b:e], concatenated_reps[1])

            # Create positive indices
            positive_indices = torch.arange(b, e).to(ap_sim.device)

            # Selectively rerank only hard negatives
            ap_scores = self.call_reranker_selective(
                anchor_texts, positive_texts, ap_sim, positive_indices
            )
            aa_scores = self.call_reranker_selective(
                anchor_texts, all_anchor_texts, aa_sim, positive_indices
            )
            pp_scores = self.call_reranker_selective(
                positive_texts[b:e], positive_texts, pp_sim,
                torch.arange(len(positive_texts[b:e])).to(pp_sim.device)
            )

            # Define the anchor threshold (diagonal scores)
            threshold_scores = ap_scores.diagonal(offset=b).view(-1, 1)

            # Apply false negative suppression
            def mask_false_negatives(scores, sim_mat, positive_mask: Tensor | None = None):
                if self.margin_strategy == "absolute":
                    mask = scores > (threshold_scores - self.margin)
                elif self.margin_strategy == "relative":
                    mask = scores > (threshold_scores * (1 - self.margin))

                if positive_mask is not None:
                    mask = mask & ~positive_mask
                sim_mat[mask] = -torch.inf
                return sim_mat

            # Create positive mask
            positive_mask = torch.eye(*ap_scores.shape, dtype=torch.bool, device=ap_scores.device)
            positive_mask = positive_mask.roll(b)

            # Apply filtering
            ap_sim = mask_false_negatives(ap_scores, ap_sim, positive_mask=positive_mask)
            aa_sim = mask_false_negatives(aa_scores, aa_sim)
            pp_sim = mask_false_negatives(pp_scores, pp_sim)

            # Concatenate similarities
            scores = torch.cat([ap_sim, aa_sim, pp_sim], dim=1)

            # Handle additional negatives if present
            if len(concatenated_reps) > 2:
                for i in range(2, len(concatenated_reps)):
                    neg_texts = concatenated_texts[i]
                    neg_sim = self.sim_matrix(concatenated_reps[0][b:e], concatenated_reps[i])
                    neg_scores = self.call_reranker_selective(
                        anchor_texts, neg_texts, neg_sim, positive_indices
                    )
                    neg_sim = mask_false_negatives(neg_scores, neg_sim)
                    scores = torch.cat([scores, neg_sim], dim=1)

            # Calculate loss
            scores = scores / self.temperature
            loss_mbatch: torch.Tensor = self.cross_entropy_loss(scores, labels[b:e]) * len(scores) / batch_size
            if with_backward:
                loss_mbatch.backward()
                loss_mbatch = loss_mbatch.detach()
            losses.append(loss_mbatch)

        loss = sum(losses)
        return loss

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        # Step (1): Extract texts from input features for reranker
        all_texts = []
        for sentence_feature in sentence_features:
            texts = self.tokenizer.batch_decode(sentence_feature["input_ids"], skip_special_tokens=True)
            all_texts.append(texts)

        # Step (2): A quick embedding step without gradients/computation graphs to get all the embeddings
        reps = []
        self.random_states = []
        texts_by_batch = []
        for i, sentence_feature in enumerate(sentence_features):
            reps_mbs = []
            random_state_mbs = []
            texts_mbs = []
            start_idx = 0
            for reps_mb, random_state in self.embed_minibatch_iter(
                sentence_feature=sentence_feature,
                with_grad=False,
                copy_random_state=True,
            ):
                reps_mbs.append(reps_mb.detach().requires_grad_())
                random_state_mbs.append(random_state)
                # Keep track of texts for each minibatch
                end_idx = start_idx + len(reps_mb)
                texts_mbs.append(all_texts[i][start_idx:end_idx])
                start_idx = end_idx
            reps.append(reps_mbs)
            self.random_states.append(random_state_mbs)
            texts_by_batch.append(texts_mbs)

        if torch.is_grad_enabled():
            # Step (3): Calculate the loss with selective reranking
            loss = self.calculate_loss_and_cache_gradients(reps, texts_by_batch)

            # Step (4): Connect cached gradients via backward hook
            loss.register_hook(partial(_backward_hook, sentence_features=sentence_features, loss_obj=self))
        else:
            loss = self.calculate_loss(reps, texts_by_batch)
        return loss

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "reranker_url": self.reranker_url,
            "temperature": self.temperature,
            "mini_batch_size": self.mini_batch_size,
            "margin_strategy": self.margin_strategy,
            "margin": self.margin,
            "reranker_batch_size": self.reranker_batch_size,
            "instruction": self.instruction,
            "max_length": self.max_length,
            "timeout": self.timeout,
            "use_simple_batch": self.use_simple_batch,
            "similarity_threshold": self.similarity_threshold,
            "top_k_ratio": self.top_k_ratio,
            "min_top_k": self.min_top_k,
            "max_candidates": self.max_candidates,
            "hybrid_threshold": self.hybrid_threshold,
        }
