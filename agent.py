# Standard Library Imports
import sys
import asyncio
import logging
import os
import datetime
from typing import TypedDict
from dotenv import load_dotenv

# LangGraph & LangChain Imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

# MCP Client Imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Custom Imports (Schemas & Outputs)
from schemas import SearchStrategy, TripItinerary
from outputs import save_itinerary_to_markdown

# --- 1. Configuration & Logging Setup ---

# Configure logging to standard output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("TravelAgent")

# Load environment variables
load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    logger.error("GOOGLE_API_KEY not found in environment variables.")
    raise ValueError("GOOGLE_API_KEY not found. Please check your .env file.")

# Initialize LLM with strict temperature for consistency
llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)

# --- 2. State Definition ---

class AgentState(TypedDict):
    """
    Maintains the state of the agent workflow.
    """
    user_request: str
    search_strategy: SearchStrategy
    search_results: str
    final_itinerary: TripItinerary

# --- 3. Node Definitions ---

def planner_node(state: AgentState):
    """
    Generates a search strategy based on user request and current date.
    Does not execute the search, only plans the queries.
    """
    logger.info("Planner Node: Generating search strategy.")
    
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # Configure LLM to force structured output based on SearchStrategy schema
    planner = llm.with_structured_output(SearchStrategy)
    
    system_prompt = (
        f"You are a Travel Architect & Logistics Expert. "
        f"Current Date: {current_date}.\n\n"
        "Your goal is to design a targeted information retrieval strategy. "
        "Do not generate the itinerary yet. Focus purely on HOW to find the best data.\n\n"
        "Guidelines:\n"
        "1. Use search-engine friendly keywords.\n"
        "2. Plan queries for attractions, hidden gems, and logistics.\n"
        "3. Target specific historical eras or sites if requested.\n\n"
        "Explain in 'reasoning' why these searches are necessary."
    )
    
    strategy = planner.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["user_request"])
    ])
    
    return {"search_strategy": strategy}

async def executor_node(state: AgentState):
    """
    Executes the planned queries using the MCP Server via Stdio protocol.
    Spawns a subprocess for the server to ensure isolation.
    """
    logger.info("Executor Node: Connecting to MCP Server.")
    
    queries = state["search_strategy"].queries
    
    # Define server execution parameters (subprocess)
    server_params = StdioServerParameters(
        command=sys.executable, 
        args=["mcp_server.py"], 
        env=None 
    )

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # Verify connection by listing tools (debug level only)
                tools = await session.list_tools()
                logger.debug(f"Connected. Tools: {[t.name for t in tools.tools]}")
                
                # Execute the search tool
                logger.info(f"Executing search for {len(queries)} queries.")
                result = await session.call_tool("search_tourism", arguments={"queries": queries})
                
                # Extract and concatenate text content from results
                final_text = ""
                for content in result.content:
                    if content.type == "text":
                        final_text += content.text
                
                logger.info("Search completed successfully.")
                logger.debug(f"Raw Search Results: {final_text[:100]}...") # Log brief snippet
                
                return {"search_results": final_text}
                
    except Exception as e:
        logger.error(f"Failed to execute MCP tool: {e}")
        raise e

def writer_node(state: AgentState):
    """
    Synthesizes search results into the final TripItinerary object.
    Enforces strict adherence to the provided context.
    """
    logger.info("Writer Node: Composing final itinerary.")
    
    writer = llm.with_structured_output(TripItinerary)
    
    system_prompt = (
        "You are an expert Travel Guide. "
        "Synthesize a logical itinerary based STRICTLY on the provided search results. "
        "Ensure activities follow a realistic time flow. "
        "Do not invent specific details (like prices) if missing from context."
    )
    
    res = writer.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"User Request: {state['user_request']}\n\nSearch Context: {state['search_results']}")
    ])
    
    return {"final_itinerary": res}

# --- 4. Graph Construction ---

workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)
workflow.add_node("writer", writer_node)

# Define edges
workflow.set_entry_point("planner")
workflow.add_edge("planner", "executor")
workflow.add_edge("executor", "writer")
workflow.add_edge("writer", END)

# Configure persistence
memory = MemorySaver()

# Compile graph with interrupt before execution
app = workflow.compile(
    checkpointer=memory,
    interrupt_before=["executor"]
)

# --- 5. Main Execution ---

async def main():
    import uuid
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    logger.info(f"Starting Session ID: {thread_id}")
    
    # Input example
    user_input = (
        "I want a 3-day trip to Rio de Janeiro, Brazil. "
        "I want to AVOID the crowded beaches (like Copacabana) and focus on "
        "the bohemian culture of Santa Teresa, hiking the 'Morro Dois Irmãos' for the view, "
        "and finding an authentic 'Roda de Samba' in Lapa or Pedra do Sal at night."
    )

    logger.info(f"Processing request: {user_input}")

    # Phase 1: Planning (Run until interrupt)
    async for event in app.astream({"user_request": user_input}, config=config):
        pass 
        
    # Retrieve current state to show plan
    current_state = await app.aget_state(config)
    strategy = current_state.values["search_strategy"]
    
    # Human-in-the-loop interaction
    print(f"\n>>> APPROVAL REQUIRED <<<")
    print(f"Reasoning: {strategy.reasoning}")
    print(f"Queries: {strategy.queries}\n")
    
    approval = input("Approve Execution? (y/n): ")
    
    if approval.lower() == 'y':
        logger.info("Plan approved. Resuming execution...")
        
        # Phase 2: Execution & Writing
        result = await app.ainvoke(None, config=config)
        final_trip = result['final_itinerary']
        
        # Console Output (Preview)
        print("\n" + "="*60)
        print(f"TRIP DESTINATION: {final_trip.destination.upper()}")
        print("="*60)
        print(f"Overview: {final_trip.overview[:150]}...\n")
        print("(Generating full report...)")
        
        # File Generation (Full Itinerary)
        filename = save_itinerary_to_markdown(final_trip)
        
        if filename:
            print(f"SUCCESS! Itinerary saved to: {os.path.abspath(filename)}")

    else:
        logger.warning("Plan rejected by user. Terminating process.")

if __name__ == "__main__":
    asyncio.run(main())