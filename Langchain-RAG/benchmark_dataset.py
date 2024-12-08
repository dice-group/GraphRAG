import os
from dotenv import load_dotenv
import pandas as pd
import giskard
from openai import OpenAI
from giskard.rag import KnowledgeBase, generate_testset
from giskard.llm.client.openai import OpenAIClient
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def setup_environment():
    """Load and check environment variables"""
    load_dotenv()
    api_key = os.getenv("TENTRIS_API_KEY")
    base_url = os.getenv("TENTRIS_BASE_URL_CHAT")
    
    if not api_key or not base_url:
        raise ValueError("Missing required environment variables")
    
    return api_key, base_url

def initialize_client(api_key, base_url):
    """Initialize OpenAI client"""
    chat_client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )
    client = OpenAIClient(model='tentris', client=chat_client)
    giskard.llm.set_default_client(client)
    giskard.llm.set_embedding_model("text-embedding-3-small")

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
    df = pd.DataFrame([
        {"content": doc.page_content} for doc in documents
    ])
    return KnowledgeBase(df)

def generate_and_save_tests(knowledge_base, output_file="benchmark_dataset.json"):
    """Generate and save test set"""
    testset = generate_testset(
        knowledge_base,
        num_questions=10,
        language='en',
        agent_description='Agent helps to generate QA pairs'
    )
    testset.save(output_file)
    print(f"Test set saved to {output_file}")

def main():
    # Setup
    api_key, base_url = setup_environment()
    initialize_client(api_key, base_url)
    
    # Process document
    docs = process_document("data/speech.txt")
    
    # Create knowledge base and generate tests
    kb = create_knowledge_base(docs)
    generate_and_save_tests(kb)

if __name__ == "__main__":
    main()