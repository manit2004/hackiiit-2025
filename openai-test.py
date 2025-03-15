"""
Interactive chat example using OpenAI agent with conversation memory.
"""

import os
import random
from moya.conversation.thread import Thread
from moya.tools.base_tool import BaseTool
from moya.tools.ephemeral_memory import EphemeralMemory
from moya.tools.tool_registry import ToolRegistry
from moya.registry.agent_registry import AgentRegistry
from moya.orchestrators.simple_orchestrator import SimpleOrchestrator
from moya.agents.azure_openai_agent import AzureOpenAIAgent, AzureOpenAIAgentConfig
from moya.conversation.message import Message
import dotenv
dotenv.load_dotenv()
import pandas as pd
from io import StringIO
from moya.tools.base_tool import BaseTool

def setup_memory_components():
    """Set up memory components for the agents."""
    tool_registry = ToolRegistry()
    EphemeralMemory.configure_memory_tools(tool_registry)
    return tool_registry

def import_to_csv(data:str) -> str:
    """
    Converts a comma-separated string into a pandas DataFrame and saves it as a CSV file.

    Args:
        data (str): The CSV-formatted string (e.g., Day,Breakfast,Lunch,Dinner,...).

    Returns:
        str: Confirmation message indicating successful saving.
    """
    print(data)
    csv_buffer = StringIO(data)
    df = pd.read_csv(csv_buffer)
    df.to_csv("meal_plan.csv", index=False)
    return f"CSV data successfully saved to {"meal_plan.csv"}"

def import_to_csv_tool():
    import_to_csv_tool = BaseTool(
        name="import_to_csv",
        description="Converts a comma-separated string into a pandas DataFrame and saves it as a CSV file.",
        function=import_to_csv,
        parameters={
            "data": {
                "type": "string",
                "description": "The CSV-formatted string (e.g., Day,Breakfast,Lunch,Dinner,...)."
            }
        },
        required=["data"]
        )
    return import_to_csv_tool

def create_onboarding_agent(tool_registry):
    onboarding_agent_config = AzureOpenAIAgentConfig(
        agent_name="chat_agent",
        description="An AI-powered chat agent for onboarding users to the AI-Powered Grocery Manager which makes meal plans based on user preferences.",
        model_name="gpt-4o",
        agent_type="ChatAgent",
        tool_registry=tool_registry,
        system_prompt="""
            ### System Prompt for Onboarding AI Agent

            **Role and Objective:**
            You are an intelligent Onboarding AI Agent for the AI-Powered Grocery Manager, developed by Team Binary Brains. Your goal is to efficiently gather comprehensive user data to enable personalized grocery management and consumption tracking.

            ### Instructions for the AI Agent:

            1. **User Interaction and Data Collection:**
            - Engage users with a friendly and conversational tone.
            - Ask step-by-step questions to gather data on:
                - Household size and number of members.
                - Number of meals per day.
                - Food preferences and dietary restrictions.
                - Specific meal choices for different days and times (e.g., Sunday dinner).
                - Dietary goals (e.g., weight loss, muscle gain, balanced nutrition).

            2. **Structured Data Generation:**
            - Organize the collected data into a comma-separated string format where:
                ```
                Day,Breakfast,Lunch,Dinner
                Monday,Oatmeal,Grilled Chicken Salad,Pasta
                Tuesday,Smoothie,Quinoa Bowl,Stir Fry
                Wednesday,Pancakes,Burrito Bowl,Grilled Salmon
                Thursday,Avocado Toast,Caesar Salad,Spaghetti
                Friday,Fruit Smoothie,Sushi,BBQ Ribs
                Saturday,French Toast,Veggie Wrap,Pizza
                Sunday,Eggs Benedict,Chicken Curry,Steak
                ```

            3. **User-Friendly Feedback and Validation:**
            - Confirm collected data with the user.
            - Allow users to make modifications and refine their preferences.

            4. **Data Handling and Privacy:**
            - Ensure data security and privacy compliance.
            - Store data in a format compatible with the AI-Powered Grocery Manager’s system for future analysis and optimization.

            5. **Seamless Integration with Other Agents:**
            - Pass the collected data to the Smart Shopping List Generator Agent and Consumption Tracking Agent.
            - Assist the Expiration & Freshness Monitoring Agent in tracking perishable items.

            6. **Error Handling and Adaptive Learning:**
            - Handle incomplete or unclear responses by asking follow-up questions.
            - Learn from user feedback to improve future interactions.

            7. **CSV Conversion Upon User Confirmation:**
            - When the user confirms that the data is final, call the `import_to_csv` tool with the comma-separated string format and convert it into a pandas DataFrame to save as a CSV file in the local directory.

            Your ultimate goal is to make the user's grocery management experience seamless, efficient, and personalized.
        """,
        api_key=os.getenv("OPENAI_API_KEY"),
        api_base=os.getenv("API_ENDPOINT") ,  # Use default OpenAI API base
        api_version=os.getenv("AZURE_OPENAI_API_VERSION") or "2024-12-01-preview",
        organization=None  # Use default organization
    )

    # Create Azure OpenAI agent with memory capabilities
    onboarding_agent = AzureOpenAIAgent(
        config=onboarding_agent_config
    )

    return onboarding_agent

def setup_agent():
    """
    Set up the AzureOpenAI agent with memory capabilities and return the orchestrator and agent.

    Returns:
        tuple: A tuple containing the orchestrator and the agent.
    """
    tool_registry = setup_memory_components()
    tool_registry.register_tool(import_to_csv_tool())
    agent_registry = AgentRegistry()
    onboarding_agent=create_onboarding_agent(tool_registry)
    agent_registry.register_agent(onboarding_agent)
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
