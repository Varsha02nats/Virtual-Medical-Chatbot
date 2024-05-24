from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

# Define global variables for docsearch and chain
docsearch = None
chain = None

def setup():
    # Load environment variables from .env file
    load_dotenv()

    # Download Hugging Face embeddings
    embeddings = download_hugging_face_embeddings()

    # Initialize the FAISS index
    index_path = "faiss_index"
    global docsearch
    docsearch = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    # Define prompt template for bot
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Get Google API key
    google_api_key = os.environ.get('GOOGLE_API_KEY')

    # Initialize Google Generative AI model
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=google_api_key)

    # Load QA chain
    global chain
    chain = load_qa_chain(model, chain_type="stuff", prompt=PROMPT)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    global docsearch, chain
    msg = request.form["msg"]
    docs = docsearch.similarity_search(msg)
    result = chain.invoke({"input_documents": docs, "question": msg}, return_only_outputs=True)
    answer = result.get('output_text', 'No answer found')  # Get the answer or default to 'No answer found' if not present
    return answer

if __name__ == '__main__':
    setup()
    app.run(debug=True)
