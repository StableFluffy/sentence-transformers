from __future__ import annotations

import asyncio
from collections.abc import Iterable, Iterator
from contextlib import nullcontext
from functools import partial
from typing import Any, Literal

import aiohttp
import numpy as np
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
    loss_obj: RerankerCachedGISTEmbedLoss,
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


class RerankerCachedGISTEmbedLoss(nn.Module):
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
    ) -> None:
        """
        This loss combines :class:`CachedGISTEmbedLoss` with a reranker API server instead of a guide model.
        It uses an external reranker to score query-document pairs and filter false negatives, while maintaining
        the memory efficiency of gradient caching.

        The reranker provides relevance scores which are used to:
        - Filter out hard negatives that are actually relevant (false negatives)
        - Guide the contrastive learning process for better in-batch negative selection

        Args:
            model: SentenceTransformer model to train
            reranker_url: URL of the reranker API server (e.g., "http://localhost:8000")
            temperature: Temperature parameter to scale the cosine similarities
            mini_batch_size: Mini-batch size for the forward pass
            show_progress_bar: If True, show progress bar during training
            margin_strategy: Strategy for false negative filtering ("absolute" or "relative")
            margin: Margin value for filtering negatives
            reranker_batch_size: Maximum batch size for each reranker API call. The loss function
                will automatically split larger requests into multiple batches of this size and
                combine them into a single batch API request for efficiency.
            instruction: Task instruction for the reranker
            max_length: Maximum token length for reranker
            timeout: Timeout for API calls in seconds
            use_simple_batch: If True, uses /rerank_batch_simple endpoint which assumes all
                requests have the same instruction/max_length for maximum vLLM efficiency.
                If False, uses /rerank_batch which handles mixed parameters but may be slower.

        Example:
            ::

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
                from datasets import Dataset

                model = SentenceTransformer("microsoft/mpnet-base")
                train_dataset = Dataset.from_dict({
                    "anchor": ["It's nice weather outside today.", "He drove to work."],
                    "positive": ["It's so sunny.", "He took the car to the office."],
                })
                loss = losses.RerankerCachedGISTEmbedLoss(
                    model,
                    reranker_url="http://localhost:8000",
                    mini_batch_size=64,
                    margin_strategy="absolute",
                    margin=0.1
                )

                trainer = SentenceTransformerTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()
        """
        super().__init__()
        if isinstance(model[0], StaticEmbedding):
            raise ValueError(
                "RerankerCachedGISTEmbedLoss is not compatible with a SentenceTransformer model based on a StaticEmbedding."
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

    def call_reranker_batch(self, queries: list[str], documents: list[str]) -> np.ndarray:
        """
        Call reranker API to get scores for query-document pairs.
        Returns a matrix of scores where scores[i, j] is the relevance of document j to query i.
        """
        # Create all query-document pairs
        pairs = []
        for query in queries:
            for doc in documents:
                pairs.append((query, doc))

        # Split into batches and prepare requests
        batch_requests = []
        for i in range(0, len(pairs), self.reranker_batch_size):
            batch = pairs[i : i + self.reranker_batch_size]
            batch_requests.append({
                "pairs": batch,
                "instruction": self.instruction,
                "max_length": self.max_length,
            })

        # Process using batch API
        async def process_batch_api():
            async with aiohttp.ClientSession() as session:
                # Choose endpoint based on configuration
                endpoint = "/rerank_batch_simple" if self.use_simple_batch else "/rerank_batch"
                
                # Call batch API endpoint
                async with session.post(
                    f"{self.reranker_url}{endpoint}",
                    json=batch_requests,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        # Flatten all scores from batch responses
                        all_scores = []
                        for batch_response in data:
                            all_scores.extend(batch_response["scores"])
                        return all_scores
                    else:
                        # Fallback to individual batch processing if batch API not available
                        tasks = []
                        for req in batch_requests:
                            tasks.append(self._call_reranker_async(session, req["pairs"]))
                        batch_results = await asyncio.gather(*tasks)
                        return [score for batch in batch_results for score in batch]

        # Run the async function
        scores = asyncio.run(process_batch_api())

        # Reshape to matrix form
        scores_matrix = np.array(scores).reshape(len(queries), len(documents))
        return scores_matrix

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
        """Generalized function to calculate the cross-entropy loss and cache the gradients wrt. the embeddings."""
        loss = self.calculate_loss(reps, all_texts, with_backward=True)
        loss = loss.detach().requires_grad_()

        self.cache = [[r.grad for r in rs] for rs in reps]

        return loss

    def calculate_loss(self, reps: list[list[Tensor]], all_texts: list[list[str]], with_backward: bool = False) -> Tensor:
        """Calculate the cross-entropy loss using reranker scores for false negative filtering."""
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

            # Get reranker scores
            # For anchor-positive pairs
            ap_reranker_scores = self.call_reranker_batch(anchor_texts, positive_texts)
            # For anchor-anchor pairs
            aa_reranker_scores = self.call_reranker_batch(anchor_texts, all_anchor_texts)
            # For positive-positive pairs
            pp_texts_mb = positive_texts[b:e]
            pp_reranker_scores = self.call_reranker_batch(pp_texts_mb, positive_texts)

            # Convert reranker scores to tensors
            ap_reranker_scores = torch.tensor(ap_reranker_scores, device=concatenated_reps[0].device)
            aa_reranker_scores = torch.tensor(aa_reranker_scores, device=concatenated_reps[0].device)
            pp_reranker_scores = torch.tensor(pp_reranker_scores, device=concatenated_reps[0].device)

            # Define the anchor threshold for each similarity matrix (diagonal scores)
            reranker_threshold = ap_reranker_scores.diagonal(offset=b).view(-1, 1)

            # Compute similarity scores for the current mini-batch
            ap_sim = self.sim_matrix(concatenated_reps[0][b:e], concatenated_reps[1])  # anchor-positive similarity
            aa_sim = self.sim_matrix(concatenated_reps[0][b:e], concatenated_reps[0])  # anchor-anchor similarity
            pp_sim = self.sim_matrix(concatenated_reps[1][b:e], concatenated_reps[1])  # positive-positive similarity

            # This uses reranker scores as a dynamic threshold to identify and suppress false negatives
            def mask_false_negatives(reranker_scores, sim_mat, positive_mask: Tensor | None = None):
                if self.margin_strategy == "absolute":
                    # Remove samples whose reranker score is higher than (positive_score - margin)
                    mask = reranker_scores > (reranker_threshold - self.margin)
                elif self.margin_strategy == "relative":
                    # Remove samples whose reranker score is higher than (positive_score * margin)
                    mask = reranker_scores > (reranker_threshold * (1 - self.margin))

                if positive_mask is not None:
                    # Ensure true positive pairs are not masked out
                    mask = mask & ~positive_mask
                sim_mat[mask] = -torch.inf
                return sim_mat

            # Create a mask to protect true positive pairs in the anchor-positive matrix
            positive_mask = torch.eye(*ap_reranker_scores.shape, dtype=torch.bool, device=ap_reranker_scores.device)
            positive_mask = positive_mask.roll(b)

            # Apply false negative suppression to each similarity matrix using reranker scores
            ap_sim = mask_false_negatives(ap_reranker_scores, ap_sim, positive_mask=positive_mask)
            aa_sim = mask_false_negatives(aa_reranker_scores, aa_sim)
            pp_sim = mask_false_negatives(pp_reranker_scores, pp_sim)

            # Concatenate the similarity matrices
            scores = torch.cat([ap_sim, aa_sim, pp_sim], dim=1)

            # If there are negatives (len(reps) > 2), process them
            if len(concatenated_reps) > 2:
                for i in range(2, len(concatenated_reps)):  # Start from 2 since first 2 are anchor-positive
                    neg_texts = concatenated_texts[i]
                    neg_reranker_scores = self.call_reranker_batch(anchor_texts, neg_texts)
                    neg_reranker_scores = torch.tensor(neg_reranker_scores, device=concatenated_reps[0].device)
                    neg_sim = self.sim_matrix(concatenated_reps[0][b:e], concatenated_reps[i])
                    neg_sim = mask_false_negatives(neg_reranker_scores, neg_sim)
                    scores = torch.cat([scores, neg_sim], dim=1)

            # Normalize the scores and calculate the cross-entropy loss
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
        self.random_states = []  # Copy random states for exact reproduction during the second forward pass
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
            # Step (3): Calculate the loss, backward up to the embeddings and cache the gradients
            loss = self.calculate_loss_and_cache_gradients(reps, texts_by_batch)

            # Step (4): A 2nd embedding step with gradients/computation graphs and connect the cached gradients
            loss.register_hook(partial(_backward_hook, sentence_features=sentence_features, loss_obj=self))
        else:
            # If grad is not enabled (e.g. in evaluation), we don't have to worry about the gradients
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
        }
