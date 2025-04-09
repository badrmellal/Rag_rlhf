"""
DarijaBridge dataset preparation for RAG implementation.
This script downloads and prepares the DarijaBridge dataset for use in the RAG system.
"""

import os
import pandas as pd
from datasets import load_dataset
import numpy as np


def prepare_dataset(sample_size=None, min_quality=1):
    """
    Load and prepare the DarijaBridge dataset

    Parameters:
    - sample_size: Number of examples to sample (None for all)
    - min_quality: Minimum quality score to include (1 for high quality only)

    Returns:
    - Processed DataFrame, train, validation, and test splits
    """
    print("Loading DarijaBridge dataset...")
    # Load dataset from Hugging Face
    dataset = load_dataset("M-A-D/DarijaBridge")
    print(f"Dataset loaded with {len(dataset['train'])} entries")

    # Convert to pandas DataFrame
    df = pd.DataFrame(dataset['train'])
    print("Dataset preview:")
    print(df.head())

    # Check dataset structure
    print("\nDataset columns:", df.columns.tolist())
    print(f"Dataset shape: {df.shape}")

    # Filter for high-quality translations
    high_quality_df = df[df['quality'] >= min_quality].copy()
    print(f"\nFiltered for quality >= {min_quality}: {len(high_quality_df)} entries")

    # Sample if requested
    if sample_size and sample_size < len(high_quality_df):
        sampled_df = high_quality_df.sample(sample_size, random_state=42)
        print(f"Sampled {len(sampled_df)} entries for development")
    else:
        sampled_df = high_quality_df
        print(f"Using all {len(sampled_df)} high-quality entries")

    # Create a combined field for context-rich retrieval
    sampled_df['combined'] = sampled_df.apply(
        lambda row: f"English: {row['translation']} | Darija: {row['sentence']}", axis=1
    )

    # Create train/validation/test splits (80/10/10)
    train_size = int(0.8 * len(sampled_df))
    val_size = int(0.1 * len(sampled_df))

    # Shuffle the data
    shuffled_df = sampled_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split the data
    train_df = shuffled_df[:train_size]
    val_df = shuffled_df[train_size:train_size + val_size]
    test_df = shuffled_df[train_size + val_size:]

    print(f"Created splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # Save to disk
    os.makedirs('data', exist_ok=True)
    sampled_df.to_parquet('data/darija_bridge_processed.parquet')
    train_df.to_parquet('data/train.parquet')
    val_df.to_parquet('data/val.parquet')
    test_df.to_parquet('data/test.parquet')

    print("Processed data saved to 'data/' directory")

    return sampled_df, train_df, val_df, test_df


if __name__ == "__main__":
    # For quick development, use a small sample
    # Remove sample_size argument or set to None for full dataset
    prepare_dataset(sample_size=10000, min_quality=1)