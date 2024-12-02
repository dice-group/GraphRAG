# Chatbot Q&A System

A question-answering system built with LangChain, FAISS, and Streamlit that allows users to query documents using the Tentris API for embeddings and chat completion.

## Features
- Document-based question answering
- Custom embedding model integration
- Simple UI

# Quick Setup


## Installation

Install dependencies:
```bash
pip install streamlit langchain langchain-community faiss-cpu python-dotenv openai
```
## Set up environment variables
1. Navigate to the project directory.

2. Create a new .env file in the root folder (same level as main.py).

3. Add the following content to the .env file:

```bash 
TENTRIS_BASE_URL_EMBEDDINGS="http://tentris-ml.cs.upb.de:8502/v1"
TENTRIS_BASE_URL_CHAT="http://tentris-ml.cs.upb.de:8501/v1"
TENTRIS_API_KEY="your-api-key-here"
```
Replace ```your-api-key-here``` with your actual API key.


## Usage

1. Place your document in the `data/` directory as `speech.txt`

2. Run the application:
```bash
streamlit run main.py
```
3. Open your browser and go to http://localhost:8501

## How to Use
1. Type your question in the text box
2. Click 'Get Answer'
3. Wait for the response

## Testing with Benchmark Dataset
- For testing the system, you can use the provided `benchmark_dataset.json` in the directory. 
- This JSON file contains Q&A pairs specific to the DICE data for evaluating how the RAG system performs with the provided knowledge.