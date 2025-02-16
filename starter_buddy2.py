from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchResults
# import streamlit as st
from langchain_core.tools import Tool
from langchain_core.messages import SystemMessage, HumanMessage
from duckduckgo_search import DDGS
from langchain_community.document_loaders import PyPDFLoader,word_document

import os
from dotenv import load_dotenv
load_dotenv()
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

search = DuckDuckGoSearchResults(name="search",) 

tools = [search]
parser = StrOutputParser()
llm = ChatGroq(model="llama-3.3-70b-versatile")
def duckduckgo_search(query):
    """Fetches real-time search results from DuckDuckGo."""
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=3)  
    return "\n".join([f"{r['title']} - {r['href']}\nSnippet: {r['body']}" for r in results])

search_tool = Tool(
    name="DuckDuckGo Search",
    func=duckduckgo_search,
    description="Use this tool to search the internet and fetch real-time information when needed."
)
system = (
            "You are an Startup Assitant Chat bot, Answer the questions asked by the user based on the chat history"
            "The user is a software developer and trying to create a startup after leaving the job"
            "Suggest the appropriate actions to be taken by the user"   
            "Search the internet and give the best answer to the user's question"
            "Also the chat history will be given, give appropriate answer based on it"
)

prompt = ChatPromptTemplate(
    [
        ("system",system),
        ("human","{input},{history}")
    ]
)
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_chain():
    chain =llm
    document_chain=prompt | chain
    return document_chain

if __name__ == "__main__":
    get_chain()
