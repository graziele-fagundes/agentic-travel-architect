# Agentic Travel Architect

This project implements a stateful, human-in-the-loop agentic workflow for travel planning. It leverages **LangGraph** for orchestration and the **Model Context Protocol (MCP)** to decouple the reasoning engine from the execution layer (web search).

The system is designed to minimize LLM hallucinations by enforcing a strict **Plan -> Approve -> Execute** cycle, using structured outputs (Pydantic) for data validation.

## Architecture

The application is composed of two distinct processes communicating via standard input/output (stdio):

1.  **Orchestrator (Client):** A Python process running the LangGraph state machine. It handles the planner, user interaction, and final synthesis.
2.  **Tool Server (MCP):** A separate process running a FastMCP server that wraps the search provider (Tavily).

### Workflow Diagram

### Component Interaction (MCP Pattern)

The Agent spawns the MCP Server as a subprocess. This ensures isolation between the LLM logic and the external tool execution.

## Prerequisites

* Python 3.10+
* A Google AI Studio API Key (Gemini)
* A Tavily AI API Key (Search Provider)

## Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/graziele-fagundes/agentic-travel-architect.git](https://github.com/graziele-fagundes/agentic-travel-architect.git)
    cd agentic-travel-architect
    ```

2.  Create and activate a virtual environment:
    ```bash
    python -m venv venv
    # Linux/MacOS:
    source venv/bin/activate
    # Windows:
    venv\Scripts\activate
    ```

3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

This project uses environment variables for configuration.

1.  Create a `.env` file in the root directory.
2.  Add the following keys:

```ini
# Required: LLM Provider
# Get key at: [https://aistudio.google.com/api-keys](https://aistudio.google.com/api-keys)
GOOGLE_API_KEY=your_google_api_key_here

# Required: Search Provider
# Get key at: [https://app.tavily.com](https://app.tavily.com)
TAVILY_API_KEY=your_tavily_api_key_here

# Optional: LangSmith Tracing
# Get key at: [https://smith.langchain.com/](https://smith.langchain.com/)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY=your_langchain_api_key_here
LANGCHAIN_PROJECT="Agentic-Travel-Architect"

```

## Usage

Execute the main entry point:

```bash
python agent.py

```

### Operational Flow

1. **Planning Phase:** The agent generates a `SearchStrategy` based on the user request defined in `main()`.
2. **Interrupt (Human-in-the-Loop):** The graph execution pauses. The planned queries are displayed in the console.
3. **Approval:**
* Input `y`: The graph resumes. The `Executor` node sends a JSON-RPC request to the local MCP server.
* Input `n`: The process terminates.
4. **Artifact Generation:** Upon completion, a Markdown file (e.g., `Trip_to_Rio_2025.md`) is saved to the local directory `trips`.

## Project Structure

* **`agent.py`**: Main application logic. Defines the LangGraph workflow, nodes, and state management.
* **`mcp_server.py`**: Standalone server implementing the Model Context Protocol. Wraps the Tavily API.
* **`schemas.py`**: Pydantic models defining the data structures for the LLM input/output.
* **`outputs.py`**: Logic for converting structured objects into Markdown artifacts.

## Technical Implementation Details

* **Model Context Protocol (MCP):** Used to demonstrate a standardized interface for tool integration. The server runs locally but follows the same protocol used for remote MCP connections.
* **LangGraph Persistence:** Uses `MemorySaver` to persist the state between the planning phase and the execution phase, allowing for the human interrupt pattern.
* **Rate Limiting:** The `mcp_server.py` implements client-side rate limiting (max 3 queries per batch) and content truncation to manage token budgets and API quotas.