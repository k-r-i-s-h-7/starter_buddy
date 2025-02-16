import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import Tool
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, AgentType
from duckduckgo_search import DDGS

# Load API keys
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Define LLM
llm = ChatGroq(model="llama-3.3-70b-versatile")

# Context file
CONTEXT_FILE = "context.txt"

# Read & Save Context
def read_context():
    if os.path.exists(CONTEXT_FILE):
        with open(CONTEXT_FILE, "r", encoding="utf-8") as file:
            return file.read().strip()
    return ""

def save_context(context):
    with open(CONTEXT_FILE, "w", encoding="utf-8") as file:
        file.write(context)

# DuckDuckGo Search Function
def search_duckduckgo(query, max_results=5):
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=max_results)
    return "\n".join([f"{r['title']} - {r['href']}\nSnippet: {r['body']}" for r in results])

search_tool = Tool(
    name="DuckDuckGo Search",
    func=search_duckduckgo,
    description="Use this tool to fetch real-time information from the internet."
)

# Contextualizer (Summarizing and Updating Chat History)
def contextualize(history):
    system_prompt = "You are a chatbot memory manager. Summarize and maintain important context from chat history."
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{history}")
    ])
    
    contextualizer = LLMChain(llm=llm, prompt=prompt)
    return contextualizer.run({"history": history})

# Chatbot for Startups
def startup_assistant(input_text, context):
    system_prompt = (
        "You are a Startup Assistant Chatbot. The user is a software developer building a startup. "
        "Answer questions based on the chat history and decide if a search is needed."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}, {context}")
    ])
    
    assistant = LLMChain(llm=llm, prompt=prompt)
    return assistant.run({"input": input_text, "context": context})

# Decision Function: Should Search Be Used?
def should_search(input_text):
    decision_prompt = ChatPromptTemplate.from_messages([
        ("system", "Decide if web search is needed for the input.only give answer as yes or no, put no other words"),
        ("human", "{input}")
    ])
    
    decision_chain = LLMChain(llm=llm, prompt=decision_prompt)
    decision = decision_chain.run({"input": input_text}).lower()
    
    return "yes" in decision  # Search if LLM says "yes"

# --- Building the Graph with LangGraph ---
graph = StateGraph()

# Define Nodes
graph.add_node("context", lambda state: contextualize(state["history"]))
graph.add_node("should_search", lambda state: should_search(state["input"]))
graph.add_node("search", ToolNode(search_tool))
graph.add_node("chatbot", lambda state: startup_assistant(state["input"], state["context"]))

# Define Transitions
graph.add_edge("context", "should_search")
graph.add_conditional_edges(
    "should_search",
    lambda state: "search" if state["search_decision"] else "chatbot"
)
graph.add_edge("search", "chatbot")

# Entry & Exit
graph.set_entry_point("context")
graph.set_finish_point("chatbot")

# --- Run the Graph ---
executor = graph.compile()

def chat_with_agent(user_input, chat_history):
    previous_context = read_context()
    result = executor.invoke({
        "input": user_input,
        "history": previous_context
    })

    save_context(result["context"])  # Save updated context
    return result["response"]

# --- Example Usage ---
if __name__ == "__main__":
    user_input = "What are the latest AI startup trends?"
    chat_history = "User: I'm launching an AI SaaS product. Bot: What's your target market?"
    
    response = chat_with_agent(user_input, chat_history)
    print("\nAssistant Response:\n", response)
