from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def create_and_save_faiss_index(data_folder, index_path):
    # Load PDF documents
    loader = DirectoryLoader(data_folder, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(documents)

    # Download Hugging Face embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create FAISS index
    vector_store = FAISS.from_texts([chunk.page_content for chunk in text_chunks], embedding=embeddings)

    # Save FAISS index
    vector_store.save_local(index_path)

# Specify the path to the data folder and index path
data_folder = "data/"  # Folder containing PDFs
index_path = "faiss_index"  # Path to save the FAISS index

# Create and save the FAISS index
create_and_save_faiss_index(data_folder, index_path)
