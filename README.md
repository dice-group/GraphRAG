# Chatbot Q&A System

A question-answering system built with LangChain, FAISS, and Streamlit that allows users to query documents using the Tentris API for embeddings and chat completion.

## Features
- Document-based question answering
- Custom embedding model integration
- Simple UI
- Benchmark dataset generation for QA evaluation

# Quick Setup


## Installation

Install dependencies:
```bash
pip install streamlit langchain langchain-community faiss-cpu python-dotenv openai
```
## Set up environment variables
1. Navigate to the project directory.

2. Create a new `.env` file in the root folder (same level as `main.py`).

3. Add the following content to the `.env` file:

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
The `benchmark_dataset.py` script generates a test dataset to evaluate the Q&A system. It processes documents, creates a knowledge base, and generates Q&A pairs with Tentris API. 

### How Benchmark Dataset is Generated:
The **Giskard Python library** provides **[RAGET (RAG Evaluation Toolkit)](https://docs.giskard.ai/en/latest/open_source/testset_generation/testset_generation/index.html)**, which automatically generates a benchmark dataset. RAGET works by:

- Generating a list of questions, reference answers, and reference contexts directly from the knowledge base of your RAG system.
- Producing test datasets that can evaluate the retrieval, generation, and overall quality of your RAG system.

This includes simple questions, as well as more complex variations (e.g., situational, double, or conversational questions) designed to target specific components of the RAG pipeline.


### Additional Dependencies for benchmark
Install required libraries:
```bash
pip install pandas giskard
```
### Usage
1. Prepare the document: Place `speech.txt` in the `data/` folder.
2. Set environment variables like above.
3. Run the script 

### Dataset Evaluation (`dataset_eval_rag.json`)
The `dataset_eval_rag.json` file is **manually generated**, focusing on *simple node questions, multihop strategies, and some more complex queries.* This dataset provides a set of example questions along with their corresponding answers, and is useful for evaluating the performance of the RAG (Retrieval-Augmented Generation) system.
