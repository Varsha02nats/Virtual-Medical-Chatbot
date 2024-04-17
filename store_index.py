from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

# print(PINECONE_API_KEY)
# print(PINECONE_API_ENV)

data_folder = "data/"  # Specify the folder containing PDFs
extracted_data = load_pdf(data_folder)
text_chunks = text_split(extracted_data)
#print("Length of text chunks:", len(text_chunks))
embeddings = download_hugging_face_embeddings()

os.environ['PINECONE_API_KEY'] = 'xxxxxxx-xxxxxx-xxxxxx-xxxxxxx'
index_name = "chatting"
docsearch = PineconeVectorStore.from_documents(text_chunks, embeddings, index_name=index_name)
