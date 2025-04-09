# DarijaBridge RAG Translation System

This project implements a Retrieval-Augmented Generation (RAG) system for translating between English and Darija (Moroccan Arabic) using the DarijaBridge dataset.

## Project Overview

This system combines retrieval-based and generation-based approaches to improve translation quality:

1. **Retrieval**: Finding similar translation examples from the dataset
2. **Generation**: Using these examples to guide the translation process

## Setup and Installation

## Implementation Steps

1. **Clone a GitHub repository** or create a new project folder
2. Create each file with the provided code
3. Install dependencies from `requirements.txt`
4. Run the project step by step:

```bash
# First prepare the dataset
python data_preparation.py

# Then build the indices
python build_index.py

# Test the complete pipeline
python rag_pipeline.py

# Run the demo
python app.py
```

### Environment Setup

```bash
# Create a virtual environment
python -m venv venv

# Activate the environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```