from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.chains.question_answering import load_qa_chain
import os

def load_pdf(data_folder):
    loader = DirectoryLoader(data_folder, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

def text_split(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

def create_pinecone_index(text_chunks):
    os.environ['PINECONE_API_KEY'] = 'd1777631-b67a-4002-a8cc-cd6476f60d2c'
    index_name = "bot"
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    docsearch = PineconeVectorStore.from_documents(text_chunks, embeddings, index_name=index_name)
    return docsearch

def setup_conversational_chain():
    google_api_key = "AIzaSyDNq5NTSohOEl3Gy3exKFQzwv6HIItKvH0"
    prompt_template = """
    Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    Context: {context}
    Question: {question}
    
    Only return the helpful answer below and nothing else.
    Helpful answer:
    """

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=google_api_key)
    chain = load_qa_chain(model, chain_type="stuff", prompt=PROMPT)
    return chain

def main():
    # Step 1: Load PDF documents
    data_folder = "data/"  # Specify the folder containing PDFs
    extracted_data = load_pdf(data_folder)

    # Step 2: Split text into chunks
    text_chunks = text_split(extracted_data)
    print("Length of text chunks:", len(text_chunks))  # Just to check the length

    # Step 3: Download Hugging Face embeddings
    embeddings = download_hugging_face_embeddings()

    # Step 4: Create Pinecone index
    docsearch = create_pinecone_index(text_chunks)

    # Step 5: Setup conversational chain
    chain = setup_conversational_chain()

    # Step 6: User interaction
    while True:
        user_question = input("Enter your question (type 'exit' to quit): ")
        if user_question.lower() == 'exit':
            print("Exiting...")
            break
        docs = docsearch.similarity_search(user_question)
        response = chain.invoke({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        print("Question:", user_question)
        print("Answer:", response.get("output_text", "No answer found"))

if __name__ == "__main__":
    main()
