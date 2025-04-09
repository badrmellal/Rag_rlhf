"""
Building FAISS indices for the DarijaBridge dataset.
This script creates and saves vector indices for fast retrieval.
"""

import os
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pickle


def build_indices(df=None, model_name="paraphrase-multilingual-mpnet-base-v2"):
    """
    Build FAISS indices for the dataset

    Parameters:
    - df: DataFrame with the dataset (loads from disk if None)
    - model_name: Name of the SentenceTransformer model to use

    Returns:
    - Dictionary with indices and related data
    """
    # Load data if not provided
    if df is None:
        df = pd.read_parquet('data/darija_bridge_processed.parquet')
        print(f"Loaded {len(df)} entries from disk")

    # Initialize the sentence transformer model
    print("\nInitializing sentence transformer model...")
    embedder = SentenceTransformer(model_name)
    print(f"Using model: {model_name}")

    # Extract the documents
    darija_sentences = df['sentence'].tolist()
    english_sentences = df['translation'].tolist()
    combined_documents = df['combined'].tolist()

    # Create directory for indices
    os.makedirs('indices', exist_ok=True)

    # Generate embeddings for the Darija sentences
    print("Generating embeddings for Darija sentences...")
    darija_embeddings = embedder.encode(darija_sentences, show_progress_bar=True, batch_size=32)

    # Generate embeddings for English sentences
    print("Generating embeddings for English sentences...")
    english_embeddings = embedder.encode(english_sentences, show_progress_bar=True, batch_size=32)

    # Generate embeddings for the combined documents
    print("Generating embeddings for combined documents...")
    combined_embeddings = embedder.encode(combined_documents, show_progress_bar=True, batch_size=32)

    # Create FAISS indices
    print("Creating FAISS indices...")

    # For Darija sentences
    dimension = darija_embeddings.shape[1]
    index_darija = faiss.IndexFlatL2(dimension)
    index_darija.add(np.array(darija_embeddings).astype('float32'))

    # For English sentences
    dimension = english_embeddings.shape[1]
    index_english = faiss.IndexFlatL2(dimension)
    index_english.add(np.array(english_embeddings).astype('float32'))

    # For combined documents
    dimension = combined_embeddings.shape[1]
    index_combined = faiss.IndexFlatL2(dimension)
    index_combined.add(np.array(combined_embeddings).astype('float32'))

    print(f"Created indices with {index_combined.ntotal} documents")

    # Save the indices and related data
    indices_data = {
        'darija_sentences': darija_sentences,
        'english_sentences': english_sentences,
        'combined_documents': combined_documents,
        'index_darija_binary': faiss.serialize_index(index_darija),
        'index_english_binary': faiss.serialize_index(index_english),
        'index_combined_binary': faiss.serialize_index(index_combined),
        'embedder_name': model_name
    }

    with open('indices/rag_indices.pkl', 'wb') as f:
        pickle.dump(indices_data, f)

    print("Indices saved to 'indices/rag_indices.pkl'")

    return indices_data


if __name__ == "__main__":
    build_indices()