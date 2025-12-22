import os
import logging
from typing import List
from fastmcp import FastMCP
from dotenv import load_dotenv
from tavily import TavilyClient

# --- 1. Configuration & Constants ---

# Load environment variables
load_dotenv()

# Configure logging for observability
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [MCP SERVER] - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TravelTools")

# CONSTANTS for Resource Management
MAX_QUERIES_PER_BATCH = 3 
MAX_CHARS_PER_RESULT = 300 

# Initialize FastMCP Server
mcp = FastMCP("TravelTools")

# Get API Key
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# --- 2. Tool Definitions ---

@mcp.tool()
def search_tourism(queries: List[str]) -> str:
    """
    Executes advanced web searches using Tavily AI.
    
    Args:
        queries: List of search strings. Note: Only the first 3 queries will be processed
                 to ensure performance and quota management.
    """
    # 1. Validation
    if not TAVILY_API_KEY:
        logger.error("TAVILY_API_KEY is missing. Aborting search.")
        return "Error: Server misconfigured (Missing API Key)."

    # 2. Batch Slicing
    processing_queries = queries[:MAX_QUERIES_PER_BATCH]
    logger.info(f"Processing {len(processing_queries)} queries (Requested: {len(queries)})")

    consolidated_results = []
    tavily = TavilyClient(api_key=TAVILY_API_KEY)

    # 3. Execution Loop
    for query in processing_queries:
        try:
            logger.info(f"Searching Tavily: '{query}'")
            
            # API Call Parameters:
            # - search_depth="basic": Faster and cheaper (1 credit)
            # - include_answer=True: Generates a direct answer
            response = tavily.search(
                query=query, 
                search_depth="basic", 
                max_results=2, 
                include_answer=True
            )
            
            # 4. Response Formatting
            summary = f"### Results for: '{query}'\n"
            
            # A. The AI-Generated Direct Answer
            if response.get('answer'):
                summary += f"**AI Summary**: {response['answer']}\n\n"
            
            # B. The Source Snippets (Truncated)
            for res in response.get('results', []):
                content = res.get('content', '')
                # Truncate content to save tokens
                clean_content = content[:MAX_CHARS_PER_RESULT] + "..." if len(content) > MAX_CHARS_PER_RESULT else content
                
                summary += (
                    f"- **Title**: {res.get('title', 'N/A')}\n"
                    f"  **Link**: {res.get('url', '#')}\n"
                    f"  **Content**: {clean_content}\n\n"
                )
            
            consolidated_results.append(summary)

        except Exception as e:
            logger.error(f"Search failed for '{query}': {e}")
            consolidated_results.append(f"Error searching '{query}': {str(e)}")

    # 5. Final Output
    if len(queries) > MAX_QUERIES_PER_BATCH:
         consolidated_results.append(f"\n*Note: {len(queries) - MAX_QUERIES_PER_BATCH} queries were skipped to conserve resources.*")

    return "\n".join(consolidated_results)

if __name__ == "__main__":
    mcp.run()