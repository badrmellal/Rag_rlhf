"""
Evaluation module for the RAG translation system.
This script evaluates the performance of the RAG system.
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
import nltk
from rag_pipeline import RAGPipeline


def evaluate(num_samples=100, direction="en_to_darija"):
    """
    Evaluate the RAG translation system

    Parameters:
    - num_samples: Number of samples to evaluate
    - direction: Translation direction

    Returns:
    - Evaluation metrics
    """
    # Ensure NLTK tokenizer is available
    nltk.download('punkt', quiet=True)

    # Load test data
    test_df = pd.read_parquet('data/test.parquet')
    if num_samples and num_samples < len(test_df):
        test_df = test_df.sample(num_samples, random_state=123)

    print(f"Evaluating on {len(test_df)} test samples...")

    # Initialize RAG pipeline
    pipeline = RAGPipeline()

    bleu_scores = []
    retrieval_relevance = []

    for i, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating"):
        if direction == "en_to_darija":
            source_text = row['translation']  # English
            reference = row['sentence']  # Darija
        else:
            source_text = row['sentence']  # Darija
            reference = row['translation']  # English

        # Get RAG translation
        results = pipeline.translate(source_text, direction=direction)
        generated = results['translation']

        # Calculate BLEU score
        reference_tokens = nltk.word_tokenize(reference)
        generated_tokens = nltk.word_tokenize(generated)

        # Handle empty tokens
        if not generated_tokens:
            generated_tokens = ['']

        bleu = sentence_bleu([reference_tokens], generated_tokens)
        bleu_scores.append(bleu)

        # Calculate average similarity of retrieved examples
        avg_similarity = sum(ex['similarity'] for ex in results['retrieved_examples']) / len(
            results['retrieved_examples'])
        retrieval_relevance.append(avg_similarity)

    # Calculate metrics
    metrics = {
        'average_bleu': sum(bleu_scores) / len(bleu_scores),
        'average_retrieval_relevance': sum(retrieval_relevance) / len(retrieval_relevance),
        'direction': direction,
        'num_samples': len(test_df)
    }

    # Print results
    print(f"\nEvaluation Results ({direction}, {len(test_df)} samples):")
    print(f"Average BLEU Score: {metrics['average_bleu']:.4f}")
    print(f"Average Retrieval Relevance: {metrics['average_retrieval_relevance']:.4f}")

    return metrics


if __name__ == "__main__":
    # Run evaluation on a small subset for testing
    # Increase num_samples or set to None for full evaluation
    metrics = evaluate(num_samples=20, direction="en_to_darija")