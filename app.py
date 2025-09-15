from fastapi import FastAPI
from pydantic import BaseModel
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

loader = PyPDFLoader(r"Pranav_Resume.pdf")
pages = loader.load()

splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(pages)

embedding_model = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)
vectorstore = FAISS.from_documents(docs, embedding_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

# -------
# FastAPI
# -------
app = FastAPI(title="Resume Q&A API", description="Ask questions about Pranav's resume")

class Question(BaseModel):
    query: str

@app.post("/ask")
def ask_question(payload: Question):
    result = qa_chain.invoke({"query": payload.query})
    return {"question": payload.query, "answer": result["result"]}
