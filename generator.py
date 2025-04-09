"""
Generator component for the RAG system.
This module handles the generation of translations based on retrieved examples.
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


class Generator:
    def __init__(self, model_name="Helsinki-NLP/opus-mt-en-ar"):
        """
        Initialize the generator with a translation model

        Parameters:
        - model_name: Name of the translation model to use
        """
        print(f"Loading translation model: {model_name}")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.translator = pipeline("translation", model=self.model, tokenizer=self.tokenizer)
        print("Generator initialized")

    def generate(self, text, retrieved_examples, direction="en_to_darija"):
        """
        Generate a translation based on retrieved examples

        Parameters:
        - text: Text to translate
        - retrieved_examples: List of retrieved translation examples
        - direction: Translation direction ('en_to_darija' or 'darija_to_en')

        Returns:
        - Generated translation and the prompt used
        """
        # Create prompt with examples
        prompt = f"Translate the following text:\n\n{text}\n\nHere are some similar examples:\n\n"

        for i, example in enumerate(retrieved_examples):
            if direction == "en_to_darija":
                prompt += f"Example {i + 1}:\nEnglish: {example['english']}\nDarija: {example['darija']}\n\n"
            else:
                prompt += f"Example {i + 1}:\nDarija: {example['darija']}\nEnglish: {example['english']}\n\n"

        prompt += "Translation:"

        # Use the translation model
        if direction == "en_to_darija":
            # For English to Darija
            translation = self.translator(text, max_length=128)[0]['translation_text']
        else:
            # For Darija to English
            # This is a placeholder - you might need to use a different model
            # or approach for this direction
            translation = "This direction not yet implemented"  # Placeholder

        return translation, prompt


if __name__ == "__main__":
    # Test the generator
    from retriever import Retriever

    # Initialize retriever and generator
    retriever = Retriever()
    generator = Generator()

    # Test text
    test_text = "I would like to visit Morocco next summer"
    print(f"Input text: {test_text}")

    # Retrieve examples
    retrieved_docs = retriever.retrieve(test_text, index_type='combined', top_k=3)

    # Generate translation
    translation, prompt = generator.generate(test_text, retrieved_docs, direction="en_to_darija")

    print("\nPrompt with examples:")
    print(prompt)
    print("\nGenerated translation:")
    print(translation)