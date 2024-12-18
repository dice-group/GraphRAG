import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
import openai
import os
from dotenv import load_dotenv

load_dotenv()

# Custom embeddings using Tentris endpoint
class TentrisEmbeddings(Embeddings):
    def __init__(self):
        self.client = openai.OpenAI(
            base_url=os.getenv("TENTRIS_BASE_URL_EMBEDDINGS"),
            api_key=os.getenv("TENTRIS_API_KEY"),
            timeout=60
        )
    
    def embed_documents(self, texts):
        responses = self.client.embeddings.create(input=texts, model="tentris")
        return [data.embedding for data in responses.data]
    
    def embed_query(self, text):
        response = self.client.embeddings.create(input=[text], model="tentris")
        return response.data[0].embedding

@st.cache_resource
def initialize_qa_system():
    embeddings= TentrisEmbeddings()
    index_path=("faiss_index")

    if os.path.exists(index_path):
        db=FAISS.load_local(index_path,embeddings, allow_dangerous_deserialization=True)
    else:
        # Load and split document
        loader = TextLoader("data/speech.txt")
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        db=FAISS.from_documents(docs,embeddings)
        db.save_local(index_path)
    
    # Setup chat client
    chat_client = openai.OpenAI(
        base_url=os.getenv("TENTRIS_BASE_URL_CHAT"),
        api_key=os.getenv("TENTRIS_API_KEY")
    )
    
    return db, chat_client

def get_answer(question, db, chat_client):
    # Get relevant documents
    similar_docs = db.similarity_search(question, k=5)
    
    # Prepare context and question
    context = "\n\n".join([doc.page_content for doc in similar_docs])
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    
    # Get response from chat model
    response = chat_client.chat.completions.create(
        model="tentris",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    return response.choices[0].message.content

def main():
    st.cache_resource.clear()
    st.set_page_config(page_title="Chatbot Q&A System", page_icon="🤖")
    
    st.title("📚 Chatbot Question & Answer System")
    
    # Initialize the QA system
    with st.spinner("Initializing the QA system..."):
        db, chat_client = initialize_qa_system()
    
    # Create a text input for the question
    question = st.text_input(
        "Ask a question about the document:",
        placeholder="Type your question here...",
        key="question_input"
    )
    
    # Add a submit button
    if st.button("Get Answer", type="primary"):
        if question:
            with st.spinner("Finding the answer..."):
                try:
                    answer = get_answer(question, db, chat_client)
                    
                    # Display the answer in a nice format
                    st.success("Here's what I found:")
                    st.write(answer)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a question first.")
    
    # Add some helpful information in the sidebar
    with st.sidebar:
        st.title("ℹ️ About")
        st.markdown("""
        This Q&A system allows you to ask questions about the loaded document.
        
        **How to use:**
        1. Type your question in the text box
        2. Click 'Get Answer'
        3. Wait for the response
        """)

if __name__ == "__main__":
    main()