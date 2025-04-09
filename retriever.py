"""
Retriever component for the RAG system.
This module handles retrieval of relevant documents from the FAISS indices.
"""

import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class Retriever:
    def __init__(self, indices_path='indices/rag_indices.pkl'):
        """
        Initialize the retriever with pre-built indices

        Parameters:
        - indices_path: Path to the saved indices
        """
        print("Loading indices...")
        with open(indices_path, 'rb') as f:
            indices_data = pickle.load(f)

        # Load embedder
        self.embedder_name = indices_data['embedder_name']
        self.embedder = SentenceTransformer(self.embedder_name)

        # Load documents
        self.darija_sentences = indices_data['darija_sentences']
        self.english_sentences = indices_data['english_sentences']
        self.combined_documents = indices_data['combined_documents']

        # Load indices
        self.index_darija = faiss.deserialize_index(indices_data['index_darija_binary'])
        self.index_english = faiss.deserialize_index(indices_data['index_english_binary'])
        self.index_combined = faiss.deserialize_index(indices_data['index_combined_binary'])

        print(f"Retriever initialized with {len(self.combined_documents)} documents")

    def retrieve(self, query, index_type='combined', top_k=5):
        """
        Retrieve the most similar documents to the query

        Parameters:
        - query: Query text
        - index_type: Type of index to use ('darija', 'english', or 'combined')
        - top_k: Number of documents to retrieve

        Returns:
        - List of retrieved documents with similarity scores
        """
        # Select the appropriate index and documents
        if index_type == 'darija':
            index = self.index_darija
            documents = self.darija_sentences
        elif index_type == 'english':
            index = self.index_english
            documents = self.english_sentences
        else:  # combined
            index = self.index_combined
            documents = self.combined_documents

        # Generate embedding for the query
        query_embedding = self.embedder.encode([query])

        # Search in the index
        distances, indices = index.search(np.array(query_embedding).astype('float32'), top_k)

        # Get the retrieved documents with similarity scores
        results = []
        for i, idx in enumerate(indices[0]):
            similarity = 1 / (1 + distances[0][i])  # Convert distance to similarity score

            if index_type == 'combined':
                # For combined index, extract the English and Darija sentences
                combined_doc = documents[idx]
                parts = combined_doc.split(' | ')
                english = parts[0].replace('English: ', '')
                darija = parts[1].replace('Darija: ', '')

                results.append({
                    'document': documents[idx],
                    'english': english,
                    'darija': darija,
                    'similarity': similarity,
                    'index': idx
                })
            else:
                results.append({
                    'document': documents[idx],
                    'similarity': similarity,
                    'index': idx
                })

        return results


if __name__ == "__main__":
    # Test the retriever
    retriever = Retriever()

    test_queries = [
        "Hello, how are you?",
        "I love Moroccan food",
        "Thank you very much"
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        results = retriever.retrieve(query, index_type='combined', top_k=3)

        for i, result in enumerate(results):
            print(f"Result {i + 1} (Similarity: {result['similarity']:.4f}):")
            print(f"English: {result['english']}")
            print(f"Darija: {result['darija']}")
            print("---")