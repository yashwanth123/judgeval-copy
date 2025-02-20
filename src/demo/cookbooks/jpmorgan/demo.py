from typing import Annotated

from langchain_openai import ChatOpenAI

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from typing_extensions import TypedDict
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv
from judgeval.common.tracer import Tracer, wrap, JudgevalCallbackHandler
from judgeval.scorers import FaithfulnessScorer, AnswerRelevancyScorer, AnswerCorrectnessScorer

import asyncio
import os
from typing import Any, Optional
from uuid import UUID

import openai
import os
import asyncio
from tavily import TavilyClient
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

from vectordbdocs import data

judgment = Tracer(api_key=os.getenv("JUDGMENT_API_KEY"))

# Define our state type
class AgentState(TypedDict):
    messages: list[BaseMessage]
    category: Optional[str]
    
def populate_vector_db(collection, data):
    """
    Populate the vector DB with financial information.
    """
    for data in data:
        collection.add(
            documents=[data['information']],
            metadatas=[{"category": data['category']}],
            ids=[f"category_{data['category'].lower().replace(' ', '_')}"]
        )

# Define a ChromaDB collection for document storage
client = chromadb.Client()
collection = client.create_collection(
    name="financial_docs",
    embedding_function=embedding_functions.OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_API_KEY"))
)


populate_vector_db(collection, data)


# Replace placeholder retrievers with actual document retrieval
def pnl_retriever(state: AgentState) -> AgentState:
    query = state["messages"][-1].content
    results = collection.query(
        query_texts=[query],
        where={"category": "pnl"},
        n_results=3
    )
    return {"messages": state["messages"], "documents": results["documents"][0]}

def balance_sheet_retriever(state: AgentState) -> AgentState:
    query = state["messages"][-1].content
    results = collection.query(
        query_texts=[query],
        where={"category": "balance_sheets"},
        n_results=3
    )
    return {"messages": state["messages"], "documents": results["documents"][0]}

def stock_retriever(state: AgentState) -> AgentState:
    query = state["messages"][-1].content
    results = collection.query(
        query_texts=[query],
        where={"category": "stocks"},
        n_results=3
    )
    return {"messages": state["messages"], "documents": results["documents"][0]}

async def main():
    with judgment.trace(
        "langgraph_run1",
        project_name="langgraph_basic",
        overwrite=True
    ) as trace:

        # Initialize the graph
        graph_builder = StateGraph(AgentState)
        
        # Create the classifier node with a system prompt
        def classify(state: AgentState) -> AgentState:
            messages = state["messages"]
            response = ChatOpenAI(model="gpt-4o-mini", temperature=0).invoke(
                input=[
                    SystemMessage(content="""You are a financial query classifier. Your job is to classify user queries into one of three categories:
                    - 'pnl' for Profit and Loss related queries
                    - 'balance_sheets' for Balance Sheet related queries
                    - 'stocks' for Stock market related queries
                    
                    Respond ONLY with the category name in lowercase, nothing else."""),
                    *messages
                ]
            )
            return {"messages": state["messages"], "category": response.content}

        # Add classifier node
        graph_builder.add_node("classifier", classify)
        
        # Add router node to direct flow based on classification
        def router(state: AgentState) -> str:
            return state["category"]
        
        # Add conditional edges based on classification
        graph_builder.add_conditional_edges(
            "classifier",
            router,
            {
                "pnl": "pnl_retriever",
                "balance_sheets": "balance_sheet_retriever",
                "stocks": "stock_retriever"
            }
        )
        
        # Add retriever nodes (placeholder functions for now)
        graph_builder.add_node("pnl_retriever", pnl_retriever)
        graph_builder.add_node("balance_sheet_retriever", balance_sheet_retriever)
        graph_builder.add_node("stock_retriever", stock_retriever)
        
        # Add response generator node
        def generate_response(state: AgentState) -> AgentState:
            messages = state["messages"]
            documents = state.get("documents", "")
            
            response = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0).invoke(
                input=[
                    SystemMessage(content=f"""You are a financial assistant. Use the following context to create a SQL query to retrieve the data from the database:
                                  
                    You can use any table schema you want.
                    
                    Context: {documents}
                    
                    If you cannot answer the question based on the context, say so."""),
                    *messages
                ]
            )
            
            return {"messages": messages + [response], "documents": documents}

        # Add edges from retrievers to response generator
        graph_builder.add_node("response_generator", generate_response)
        graph_builder.add_edge("pnl_retriever", "response_generator")
        graph_builder.add_edge("balance_sheet_retriever", "response_generator")
        graph_builder.add_edge("stock_retriever", "response_generator")
        
        graph_builder.set_entry_point("classifier")
        graph_builder.set_finish_point("response_generator")

        # Compile the graph
        graph = graph_builder.compile()
        
    response = await graph.ainvoke({
        "messages": [HumanMessage(content="Please calculate our PNL on Apple stock. We have 100 shares, we bought at $100, it is now at $200.")],
        "category": None
    })
    
    print(f"Response: {response=}")

if __name__ == "__main__":
    asyncio.run(main())
