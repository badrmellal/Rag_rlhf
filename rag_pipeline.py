"""
Complete RAG pipeline for translation between English and Darija.
This module integrates the retriever and generator into a complete system.
"""

from retriever import Retriever
from generator import Generator


class RAGPipeline:
    def __init__(self, retriever=None, generator=None):
        """
        Initialize the RAG pipeline with retriever and generator components

        Parameters:
        - retriever: Retriever instance (will create one if None)
        - generator: Generator instance (will create one if None)
        """
        # Initialize components if not provided
        self.retriever = retriever if retriever else Retriever()
        self.generator = generator if generator else Generator()
        print("RAG Pipeline initialized")

    def translate(self, text, direction="en_to_darija", top_k=3):
        """
        Translate text using the RAG pipeline

        Parameters:
        - text: Text to translate
        - direction: Translation direction ('en_to_darija' or 'darija_to_en')
        - top_k: Number of examples to retrieve

        Returns:
        - Results including translation and retrieved examples
        """
        # Determine source and target languages based on direction
        if direction == "en_to_darija":
            query = text  # English text
            index_type = 'combined'
        else:
            query = text  # Darija text
            index_type = 'combined'

        # Step 1: Retrieve relevant examples
        retrieved_docs = self.retriever.retrieve(query, index_type=index_type, top_k=top_k)

        # Step 2: Generate translation
        translation, prompt = self.generator.generate(text, retrieved_docs, direction=direction)

        # Return all results
        return {
            'input_text': text,
            'translation': translation,
            'retrieved_examples': retrieved_docs,
            'prompt': prompt,
            'direction': direction
        }


if __name__ == "__main__":
    # Test the RAG pipeline
    pipeline = RAGPipeline()

    test_texts = [
        "Good morning, how are you today?",
        "I would like to learn Moroccan Arabic",
        "Where is the nearest restaurant?"
    ]

    for text in test_texts:
        print(f"\nInput: {text}")
        results = pipeline.translate(text, direction="en_to_darija")

        print(f"Translation: {results['translation']}")
        print("\nRetrieved Examples:")
        for i, example in enumerate(results['retrieved_examples']):
            print(f"Example {i + 1} (Similarity: {example['similarity']:.4f}):")
            print(f"English: {example['english']}")
            print(f"Darija: {example['darija']}")
            print("---")