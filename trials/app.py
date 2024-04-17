from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone
import pinecone
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')


embeddings = download_hugging_face_embeddings()

#Initializing the Pinecone
os.environ['PINECONE_API_KEY'] = 'd1777631-b67a-4002-a8cc-cd6476f60d2c'
index_name = "chatting"

#Loading the index
docsearch=Pinecone.from_existing_index(index_name, embeddings)


PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

google_api_key = os.environ.get('google_api_key')

model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=google_api_key)

chain = load_qa_chain(model, chain_type="stuff", prompt=PROMPT)

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    docs = docsearch.similarity_search(input)
    result = chain.invoke({"input_documents": docs, "question": input}, return_only_outputs=True)
    return str(result["result"])

# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port= 8080, debug= True)

if __name__ == '__main__':
    app.run(debug= True)






# @app.route("/get", methods=["GET", "POST"])
# def chat():
#     msg = request.form["msg"]
#     input = msg
#     print(input)
#     result=chain({"query": input})
#     print("Response : ", result["result"])
#     return str(result["result"])




