"""
Interactive chat example using OpenAI agent with conversation memory.
"""

import os
import re
import csv
from datetime import datetime, timedelta
import random
from moya.conversation.thread import Thread
from moya.tools.base_tool import BaseTool
from moya.tools.ephemeral_memory import EphemeralMemory
from moya.tools.tool_registry import ToolRegistry
from moya.registry.agent_registry import AgentRegistry
from moya.orchestrators.simple_orchestrator import SimpleOrchestrator
from moya.agents.azure_openai_agent import AzureOpenAIAgent, AzureOpenAIAgentConfig
from moya.conversation.message import Message

def setup_memory_components():
    """Set up memory components for the agents."""
    tool_registry = ToolRegistry()
    EphemeralMemory.configure_memory_tools(tool_registry)
    return tool_registry

def product_inventory_query(query: str) -> str:
    """
    Reads the product database from a CSV file and answers queries related to product expiry.

    Supported queries:
      - "how many products will expire in X day(s)" (exactly X days)
      - "list all products and in how many days they will expire"
      - "how many products are expired"
      - "which products are fresh"

    Args:
        query (str): The user's natural language query.

    Returns:
        str: An answer based on the product data.
    """
    # Load product data from CSV
    products = []
    try:
        with open("product_db.csv", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                row["quantity"] = int(row["quantity"])
                products.append(row)
    except FileNotFoundError:
        return "Product database not found. Please ensure 'product_db.csv' exists in the working directory."

    # For demonstration, use a fixed current date.
    # In a real application, you might use: current_date = datetime.today()
    current_date = datetime.strptime("2025-03-16", "%Y-%m-%d")
    query_lower = query.lower().strip()

    # Branch 1: Query for products expiring in exactly X days
    match = re.search(r"expire in (\d+)\s*day[s]?", query_lower)
    if match:
        days = int(match.group(1))
        count = 0
        expiring_products = []
        for product in products:
            expiry_date = datetime.strptime(product["expiry_date"], "%Y-%m-%d")
            days_until_expiry = (expiry_date - current_date).days
            # Only include products that expire exactly in the specified number of days
            if days_until_expiry == days:
                count += product["quantity"]
                expiring_products.append(
                    f"{product['product_name']} (Quantity: {product['quantity']}, Expires in: {days_until_expiry} day{'s' if days_until_expiry != 1 else ''})"
                )
        if expiring_products:
            product_list_str = "\n".join(f"- {item}" for item in expiring_products)
            return f"There are {count} units of products that will expire in exactly {days} day{'s' if days != 1 else ''}. Here is the list:\n{product_list_str}"
        else:
            return f"No products will expire in exactly {days} day{'s' if days != 1 else ''}."

    # Branch 2: Query to list all products with their days until expiry
    elif "list all" in query_lower and "expire" in query_lower:
        lines = []
        for product in products:
            expiry_date = datetime.strptime(product["expiry_date"], "%Y-%m-%d")
            days_until_expiry = (expiry_date - current_date).days
            lines.append(f"{product['product_name']} (Quantity: {product['quantity']}, Expires in: {days_until_expiry} day{'s' if days_until_expiry != 1 else ''})")
        if lines:
            return "Products and their expiry info:\n" + "\n".join(lines)
        else:
            return "No product data available."

    # Branch 3: Query for expired products
    elif "expired" in query_lower:
        count = 0
        for product in products:
            expiry_date = datetime.strptime(product["expiry_date"], "%Y-%m-%d")
            if expiry_date < current_date:
                count += product["quantity"]
        return f"{count} units of products are already expired."

    # Branch 4: Query for fresh products
    elif "fresh" in query_lower:
        # Define 'fresh' as having more than a threshold (e.g., 5 days) before expiry.
        threshold_days = 5
        fresh_products = set()
        for product in products:
            expiry_date = datetime.strptime(product["expiry_date"], "%Y-%m-%d")
            if (expiry_date - current_date).days > threshold_days:
                fresh_products.add(product["product_name"])
        if fresh_products:
            fresh_list = ", ".join(sorted(fresh_products))
            return f"Fresh products (with more than {threshold_days} days until expiry): {fresh_list}"
        else:
            return "No products meet the 'fresh' criteria."

    else:
        return "Query not recognized. Please ask about product expiry or freshness."

def create_product_inventory_query_tool():
    product_inventory_query_tool = BaseTool(
    name="product_inventory_query_tool",
    description="Tool to answer queries about product expiry and freshness using the product database.",
    function=product_inventory_query,
    parameters={
        "query": {
            "type": "string",
            "description": "A query about product expiry or freshness, e.g., 'How many products will expire in 7 days?'"
        }
    },
    required=["query"]
)
    return product_inventory_query_tool

def list_all_products_with_expiry() -> str:
    """
    Reads the product database from a CSV file and returns a formatted list of all products with their expiry dates.

    Returns:
        str: A formatted string listing each product with its product ID, name, buying date, expiry date, and quantity.
    """
    products = []
    try:
        with open("product_db.csv", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                row["quantity"] = int(row["quantity"])
                products.append(row)
    except FileNotFoundError:
        return "Product database not found. Please ensure 'product_db.csv' exists in the working directory."

    if not products:
        return "No product data available."

    lines = []
    for product in products:
        line = (
            f"Product ID: {product['product_id']}, "
            f"Name: {product['product_name']}, "
            f"Buying Date: {product['buying_date']}, "
            f"Expiry Date: {product['expiry_date']}, "
            f"Quantity: {product['quantity']}"
        )
        lines.append(line)

    return "Product List with Expiry Dates:\n" + "\n".join(lines)

def create_list_all_products_with_expiry_tool():
    list_products_tool = BaseTool(
    name="list_products_with_expiry_tool",
    description="Lists all products with their expiry dates and details from the product database.",
    function=list_all_products_with_expiry,
    parameters={},  # No parameters required
    required=[]
)

# Then register this tool with your tool registry:
    return list_products_tool

def get_product_details(product_query: str) -> str:
    """
    Reads the product database from a CSV file and returns details for the product(s)
    that match the provided product_query (case-insensitive).

    Args:
        product_query (str): The name (or partial name) of the product to search for.

    Returns:
        str: A formatted string containing the details of matching products, or a message if none found.
    """
    products = []
    try:
        with open("product_db.csv", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                row["quantity"] = int(row["quantity"])
                # If the product name contains the search query (case-insensitive)
                if product_query.lower() in row["product_name"].lower():
                    products.append(row)
    except FileNotFoundError:
        return "Product database not found. Please ensure 'product_db.csv' exists in the working directory."

    if not products:
        return f"No details found for product '{product_query}'."
    else:
        details_lines = []
        for product in products:
            details_lines.append(
                f"Product ID: {product['product_id']}, "
                f"Name: {product['product_name']}, "
                f"Buying Date: {product['buying_date']}, "
                f"Expiry Date: {product['expiry_date']}, "
                f"Quantity: {product['quantity']}"
            )
        return "\n".join(details_lines)

def create_get_product_details_tool():
    get_product_details_tool = BaseTool(
    name="get_product_details_tool",
    description="Retrieves details about a specific product from the product database. The search is case-insensitive.",
    function=get_product_details,
    parameters={
        "product_query": {
            "type": "string",
            "description": "The name or partial name of the product to search for."
        }
    },
    required=["product_query"]
)

# Assuming you have a tool registry already set up, register the tool:
    return get_product_details_tool


def create_inventory_agent(tool_registry):
    agent_config = AzureOpenAIAgentConfig(
        agent_name="chat_agent",
        description="An interactive chat agent",
        model_name="gpt-4o",
        agent_type="ChatAgent",
        tool_registry=tool_registry,
        system_prompt="""
            You are an interactive chat agent specializing in product inventory management.
            You have access to a product database stored in CSV format and several tools to help answer user queries:
            - Use the product_inventory_query tool to respond to questions such as "How many products will expire in X days?", "How many products are expired?", or "Which products are fresh?".
            - Use the list_all_products_with_expiry tool to list all products along with their buying dates, expiry dates, and quantities.
            - Use the get_product_details tool to retrieve detailed information about a specific product by name.
            Always begin by storing the user's message and retrieving conversation context before generating your final response.
        """,
        api_key=os.getenv("OPENAI_API_KEY"),
        api_base="https://aoi-iiit-hack-2.openai.azure.com/",  # Use default OpenAI API base
        api_version=os.getenv("AZURE_OPENAI_API_VERSION") or "2024-12-01-preview",
        organization=None  # Use default organization
    )

    # Create Azure OpenAI agent with memory capabilities
    agent = AzureOpenAIAgent(
        config=agent_config
    )
    return agent

def setup_agent():
    """
    Set up the AzureOpenAI agent with memory capabilities and return the orchestrator and agent.

    Returns:
        tuple: A tuple containing the orchestrator and the agent.
    """
    # Set up memory components
    tool_registry = setup_memory_components()
    tool_registry.register_tool(create_product_inventory_query_tool())
    tool_registry.register_tool(create_list_all_products_with_expiry_tool())
    tool_registry.register_tool(create_get_product_details_tool())


    # Set up registry and orchestrator
    agent_registry = AgentRegistry()
    inventory_agent = create_inventory_agent(tool_registry)
    agent_registry.register_agent(inventory_agent)
    orchestrator = SimpleOrchestrator(
        agent_registry=agent_registry,
        default_agent_name="chat_agent"
    )

    return orchestrator


def format_conversation_context(messages):
    """
    Format the conversation context from a list of messages.

    Args:
        messages (list): A list of message objects.

    Returns:
        str: A formatted string representing the conversation context.
    """
    context = "\nPrevious conversation:\n"
    for msg in messages:
        # Access Message object attributes properly using dot notation
        sender = "User" if msg.sender == "user" else "Assistant"
        context += f"{sender}: {msg.content}\n"
    return context


def main():
    orchestrator = setup_agent()
    thread_id = "interactive_chat_001"
    EphemeralMemory.store_message(thread_id=thread_id, sender="system", content=f"Starting conversation, thread ID = {thread_id}")

    print("Welcome to Interactive Chat! (Type 'quit' or 'exit' to end)")
    print("-" * 50)

    while True:
        # Get user input
        user_input = input("\nYou: ").strip()

        # Check for exit command
        if user_input.lower() in ['quit', 'exit']:
            print("\nGoodbye!")
            break

        # Store user message
        EphemeralMemory.store_message(thread_id=thread_id, sender="user", content=user_input)

        session_summary = EphemeralMemory.get_thread_summary(thread_id)
        enriched_input = f"{session_summary}\nCurrent user message: {user_input}"

        # Print Assistant prompt
        print("\nAssistant: ", end="", flush=True)

        # Define callback for streaming
        def stream_callback(chunk):
            print(chunk, end="", flush=True)

        # Get response using stream_callback
        response = orchestrator.orchestrate(
            thread_id=thread_id,
            user_message=enriched_input,
            stream_callback=stream_callback
        )

        # print(response)

        EphemeralMemory.store_message(thread_id=thread_id, sender="assistant", content=response)
        # Print newline after response
        print()




if __name__ == "__main__":
    main()

