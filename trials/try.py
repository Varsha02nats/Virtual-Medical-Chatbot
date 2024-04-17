from src.helper import download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone
import pinecone
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
from src.prompt import *
import os

def setup():
    # Load environment variables from .env file
    load_dotenv()

    # Get Pinecone API key and environment
    PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
    PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

    # Download Hugging Face embeddings
    embeddings = download_hugging_face_embeddings()

    # Initializing the Pinecone
    os.environ['PINECONE_API_KEY'] = 'd1777631-b67a-4002-a8cc-cd6476f60d2c'
    index_name = "chatting"

    # Loading the index
    docsearch = Pinecone.from_existing_index(index_name, embeddings)

    # Define prompt template
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Get Google API key
    google_api_key = os.environ.get('google_api_key')

    # Initialize Google Generative AI model
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=google_api_key)

    # Load QA chain
    chain = load_qa_chain(model, chain_type="stuff", prompt=PROMPT)

    return docsearch, chain

def chat(docsearch, chain):
    while True:
        user_question = input("Enter your question (type 'exit' to quit): ")
        if user_question.lower() == 'exit':
            print("Exiting...")
            break
        docs = docsearch.similarity_search(user_question)
        response = chain.invoke({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        print("Question:", user_question)
        # Print the complete answer, not relying on console width
        print("Answer:", response.get("output_text", "No answer found"))



if __name__ == '__main__':
    docsearch, chain = setup()
    chat(docsearch, chain)
