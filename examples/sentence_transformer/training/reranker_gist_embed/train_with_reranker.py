from __future__ import annotations

"""
This example shows how to train a sentence transformer model using RerankerCachedGISTEmbedLoss.
Instead of using a guide model, it uses an external reranker API to score query-document pairs
and filter false negatives during contrastive learning.

The reranker provides more accurate relevance scores compared to a simple embedding similarity,
which helps to identify and suppress false negatives more effectively.
"""

from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
from sentence_transformers.training_args import SentenceTransformerTrainingArguments


def main():
    # 1. Load a pre-trained sentence transformer model
    model = SentenceTransformer("microsoft/mpnet-base")
    
    # 2. Create a training dataset
    # In practice, you would load a real dataset with query-document pairs
    train_dataset = Dataset.from_dict({
        "anchor": [
            "What is the capital of France?",
            "How to cook pasta?",
            "What are the symptoms of COVID-19?",
            "Who wrote Romeo and Juliet?",
            "What is machine learning?",
            "How does photosynthesis work?",
            "What is the speed of light?",
            "Who painted the Mona Lisa?",
        ],
        "positive": [
            "Paris is the capital and largest city of France.",
            "To cook pasta, boil water, add salt, then add pasta and cook for 8-12 minutes.",
            "Common COVID-19 symptoms include fever, cough, and difficulty breathing.",
            "William Shakespeare wrote the play Romeo and Juliet.",
            "Machine learning is a type of artificial intelligence that enables computers to learn from data.",
            "Photosynthesis is the process by which plants convert light energy into chemical energy.",
            "The speed of light in vacuum is approximately 299,792,458 meters per second.",
            "Leonardo da Vinci painted the Mona Lisa in the early 16th century.",
        ],
    })
    
    # 3. Initialize the RerankerCachedGISTEmbedLoss
    # This loss function uses an external reranker to score query-document pairs
    loss = losses.RerankerCachedGISTEmbedLoss(
        model,
        reranker_url="http://localhost:8000",  # URL of your reranker API server
        mini_batch_size=16,  # Size of mini-batches for gradient caching
        reranker_batch_size=128,  # Batch size for reranker API calls
        margin_strategy="absolute",  # Strategy for filtering false negatives
        margin=0.1,  # Margin for false negative filtering
        temperature=0.05,  # Temperature for scaling similarities
        instruction="Given a web search query, retrieve relevant passages that answer the query",
        max_length=512,  # Maximum token length for reranker
        timeout=30,  # API timeout in seconds
        show_progress_bar=True,
    )
    
    # 4. Define training arguments
    args = SentenceTransformerTrainingArguments(
        output_dir="output/reranker-gist-embed",
        num_train_epochs=3,
        per_device_train_batch_size=32,  # This can be large due to gradient caching
        warmup_ratio=0.1,
        fp16=True,  # Enable mixed precision training
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
        save_total_limit=2,
        load_best_model_at_end=False,
        # Use NO_DUPLICATES to ensure no duplicate in-batch negatives
        batch_sampler="no_duplicates",
    )
    
    # 5. Create the trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
    )
    
    # 6. Start training
    trainer.train()
    
    # 7. Save the final model
    model.save_pretrained("output/reranker-gist-embed/final")
    
    print("Training completed!")
    
    # 8. Test the trained model
    test_queries = [
        "What is the capital of Germany?",
        "How to make coffee?",
    ]
    test_docs = [
        "Berlin is the capital of Germany.",
        "Paris is the capital of France.",
        "To make coffee, grind beans and brew with hot water.",
        "Coffee is a popular beverage.",
    ]
    
    embeddings_queries = model.encode(test_queries)
    embeddings_docs = model.encode(test_docs)
    
    # Compute similarities
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(embeddings_queries, embeddings_docs)
    
    print("\nSimilarity scores:")
    for i, query in enumerate(test_queries):
        print(f"\nQuery: {query}")
        for j, doc in enumerate(test_docs):
            print(f"  Doc {j}: {similarities[i, j]:.3f} - {doc}")


if __name__ == "__main__":
    main()