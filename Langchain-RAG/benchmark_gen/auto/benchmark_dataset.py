"""  
$ python benchmark_dataset.py [-h] [--input INPUT] [--output OUTPUT] [--num_questions NUM_QUESTIONS]

##################################################
Benchmark Dataset Generation for RAG Evaluation
##################################################

Input & Usage:
    - Input file: Text document (default: "data/speech.txt")
    - Run command: python benchmark_dataset.py
    - Optional arguments:
        --input: Path to input document
        --output: Path to output file (default: benchmark_dataset.json)
        --num_questions: Number of questions to generate (default: 10)
    - Required environment variables:
        - TENTRIS_API_KEY: API key for authentication
        - TENTRIS_BASE_URL_CHAT: Base URL for the chat API
        - TENTRIS_BASE_URL_EMBEDDINGS: Base URL for the embedding API

Computation Process:
    1. Setup API Clients (chat & embedding):
        - Validates environment variables and initializes clients for chat and embedding using LiteLLM endpoints.
        - Tests connectivity via minimal chat completion and embedding calls.

    2. Load and Split Documents:
        - Reads the specified document file into memory using TextLoader.
        - Splits the text into smaller, semantically coherent chunks using LangChain's RecursiveCharacterTextSplitter.

    3. Create Knowledge Base from Chunks:
        - Converts document chunks into a DataFrame and initializes a Giskard KnowledgeBase.
        - Computes vector embeddings for each chunk using the configured embedding model.
        - Performs topic detection via UMAP (dimensionality reduction) and HDBSCAN (clustering).
        - Generates topic names using LLM-based summarization prompts.

    4. Generate Test Questions:
        - Utilizes Giskard's question generators (e.g., simple, complex, situational) to create QA pairs:
            - Randomly selects seed documents and identifies relevant neighbors using similarity search.
            - Constructs a context string for each generator.
            - Queries the LLM to produce a question and reference answer pair.
        - Aggregates all generated QA pairs into a QATestset for evaluation.

    5. Save Test Set to JSON:
        - Serializes the QA pairs, including metadata (e.g., question type, topic, seed document ID), into a JSONL file (benchmark_dataset.json).

Output:
    - File: benchmark_dataset.json
    - Format Example:
    {
        "id": "2e4db489-118c-4100-b064-efece27eb3e8",
        "question": "What are the contact details for Dr. Yasir Mahmood?",
        "reference_answer": "Dr. Yasir Mahmood can be contacted via email at yasir.mahmood@uni-paderborn.de...",
        "reference_context": "Document 33: Hizkiel Mitiku Alemayehu is...",
        "conversation_history": [],
        "metadata": {
            "question_type": "simple",
            "seed_document_id": 33,
            "topic": "Others"
        }
    }
"""

import argparse
import os
from dotenv import load_dotenv
import litellm
import pandas as pd
import giskard
from openai import OpenAI
from giskard.rag import KnowledgeBase, generate_testset
from giskard.llm.client.openai import OpenAIClient
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# def setup_environment():
#     """Load and check environment variables"""
#     load_dotenv()
#     api_key = os.getenv("TENTRIS_API_KEY")
#     base_url = os.getenv("TENTRIS_BASE_URL_CHAT")
    
#     if not api_key or not base_url:
#         raise ValueError("Missing required environment variables")
    
#     return api_key, base_url

# def initialize_client(api_key, base_url):
#     """Initialize OpenAI client"""
#     chat_client = OpenAI(
#         base_url=base_url,
#         api_key=api_key
#     )
#     client = OpenAIClient(model='tentris', client=chat_client)
#     giskard.llm.set_default_client(client)
#     #giskard.llm.set_embedding_model("text-embedding-3-small")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate benchmark dataset for RAG evaluation')
    parser.add_argument('--input', type=str, default="data/speech.txt",
                      help='Path to input document (default: data/speech.txt)')
    parser.add_argument('--output', type=str, default="benchmark_dataset.json",
                      help='Path to output JSON file (default: benchmark_dataset.json)')
    parser.add_argument('--num_questions', type=int, default=10,
                      help='Number of questions to generate (default: 10)')
    return parser.parse_args()


def setup_clients():
    """Initialize both chat and embedding clients with LiteLLM"""
    load_dotenv()
    
    api_key = os.getenv("TENTRIS_API_KEY")
    chat_base_url = os.getenv("TENTRIS_BASE_URL_CHAT")
    embed_base_url = os.getenv("TENTRIS_BASE_URL_EMBEDDINGS")
    
    if not all([api_key, chat_base_url, embed_base_url]):
        raise ValueError("Missing required environment variables")

    # Set up chat model in Giskard using LiteLLM's OpenAI compatibility mode
    giskard.llm.set_llm_model(
        "openai/tentris",  
        api_key=api_key,
        api_base=chat_base_url
    )

    # Set up embedding model in Giskard
    giskard.llm.set_embedding_model(
        "openai/tentris",  
        api_key=api_key,
        api_base=embed_base_url
    )

    # Test connections
    try:
        # Test chat completion
        response = litellm.completion(
            model="openai/tentris",
            api_key=api_key,
            api_base=chat_base_url,
            messages=[{"role": "user", "content": "Test message"}]
        )
        print("Chat connection successful")

        # Test embedding
        response = litellm.embedding(
            model="openai/tentris",
            api_key=api_key,
            api_base=embed_base_url,
            input=["Test embedding"]
        )
        print("Embedding connection successful")

    except Exception as e:
        print(f"Connection test failed: {str(e)}")
        raise

def process_document(file_path):
    """Load and split document into chunks"""
    loader = TextLoader(file_path)
    documents = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200
    )
    
    return splitter.split_documents(documents)

def create_knowledge_base(documents):
    """Create knowledge base from document chunks"""
    df = pd.DataFrame([
        {"content": doc.page_content} for doc in documents
    ])
    return KnowledgeBase(df)

def generate_and_save_tests(knowledge_base, output_file, num_questions):
    """Generate and save test set"""
    testset = generate_testset(
        knowledge_base,
        num_questions=num_questions,
        language='en',
        agent_description='Agent helps to generate QA pairs'
    )
    testset.save(output_file)
    print(f"Test set saved to {output_file}")

def main():
    # Parse arguments
    args = parse_arguments()

    # Setup
    setup_clients()
    
    # Process document
    docs = process_document(args.input)
    
    # Create knowledge base and generate tests
    kb = create_knowledge_base(docs)
    generate_and_save_tests(kb, args.output, args.num_questions)

if __name__ == "__main__":
    main()