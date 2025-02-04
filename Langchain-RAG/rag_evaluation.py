from benchmark_gen.benchmark_dataset import parse_arguments, setup_clients, process_document, create_knowledge_base
import argparse
from giskard.rag import evaluate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import openai
import os
from langchain.embeddings.base import Embeddings
from langchain_community.document_loaders import TextLoader
from src.main import TentrisEmbeddings

def setup_rag_agent():
    # Initialize embeddings and chat client
    embeddings = TentrisEmbeddings()
    chat_client = openai.OpenAI(
        base_url=os.getenv("TENTRIS_BASE_URL_CHAT"),
        api_key=os.getenv("TENTRIS_API_KEY")
    )
    
    # Create or load FAISS index
    loader = TextLoader("data/speech.txt")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    db = FAISS.from_documents(docs, embeddings)
    
    return db, chat_client

def get_answer_fn(question: str, history=None):
    """Wrapper function for RAG agent"""
    db, chat_client = setup_rag_agent()
    
    # Format conversation history
    messages = history if history else []
    messages.append({"role": "user", "content": question})
    
    # Get relevant documents
    similar_docs = db.similarity_search(question, k=5)
    context = "\n\n".join([doc.page_content for doc in similar_docs])
    
    # Prepare prompt
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    
    # Get response from chat model
    response = chat_client.chat.completions.create(
        model="tentris",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    return response.choices[0].message.content

def main():
    # Setup clients
    args = parse_arguments()
    setup_clients(args.llm, args.embed)
    
    # Process document and create knowledge base
    docs = process_document(args.input, args.csv_source_column)
    knowledge_base = create_knowledge_base(docs)
    
    # Generate test set
    from giskard.rag import generate_testset
    testset = generate_testset(
        knowledge_base,
        num_questions=args.num_questions,
        language='en'
    )
    
    # Run evaluation
    report = evaluate(
        get_answer_fn,
        testset=testset,
        knowledge_base=knowledge_base
    )
    report.save("rag_eval_report.json")
    try:
        report.to_html("rag_eval_report.html")
        print("Evaluation report saved to rag_eval_report.html")
    except Exception as e:
        print(f"Error generating HTML report: {e}")
        # Fallback: print the report summary
        print(report)

if __name__ == "__main__":
    main()