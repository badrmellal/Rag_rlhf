"""
Gradio interface for the RAG translation system.
This script creates an interactive demo of the system.
"""

import gradio as gr
from rag_pipeline import RAGPipeline


def gradio_translate(text, direction):
    """Function to use with Gradio interface"""
    if not text.strip():
        return "Please enter text to translate", "No examples retrieved"

    # Initialize RAG pipeline
    pipeline = RAGPipeline()

    # Run the translation
    results = pipeline.translate(text, direction=direction)

    # Format examples for display
    examples_text = ""
    for i, example in enumerate(results['retrieved_examples']):
        examples_text += f"Example {i + 1} (Similarity: {example['similarity']:.4f}):\n"
        examples_text += f"English: {example['english']}\n"
        examples_text += f"Darija: {example['darija']}\n\n"

    return results['translation'], examples_text


# Create and launch Gradio interface
def create_demo():
    demo = gr.Interface(
        fn=gradio_translate,
        inputs=[
            gr.Textbox(label="Text to Translate", lines=3),
            gr.Radio(
                ["en_to_darija", "darija_to_en"],
                label="Translation Direction",
                value="en_to_darija"
            )
        ],
        outputs=[
            gr.Textbox(label="Translation", lines=3),
            gr.Textbox(label="Retrieved Examples", lines=10)
        ],
        title="DarijaBridge RAG Translation System",
        description="Translate between English and Darija using Retrieval-Augmented Generation."
    )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch()