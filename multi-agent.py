"""
Interactive chat example using OpenAI agent with conversation memory.
"""

import os
import re
import csv
from datetime import datetime, timedelta
import random
import openai
import pandas as pd
from io import StringIO
from moya.classifiers.llm_classifier import LLMClassifier
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

# def product_inventory_query(query: str) -> str:
#     """
#     Reads the product database from a CSV file and answers queries related to product expiry.

#     Supported queries:
#       - "how many products will expire in X day(s)" (exactly X days)
#       - "list all products and in how many days they will expire"
#       - "how many products are expired"
#       - "which products are fresh"

#     Args:
#         query (str): The user's natural language query.

#     Returns:
#         str: An answer based on the product data.
#     """
#     # Load product data from CSV
#     products = []
#     try:
#         with open("product_db.csv", newline="") as csvfile:
#             reader = csv.DictReader(csvfile)
#             for row in reader:
#                 row["quantity"] = int(row["quantity"])
#                 products.append(row)
#     except FileNotFoundError:
#         return "Product database not found. Please ensure 'product_db.csv' exists in the working directory."

#     # For demonstration, use a fixed current date.
#     # In a real application, you might use: current_date = datetime.today()
#     current_date = datetime.strptime("2025-03-16", "%Y-%m-%d")
#     query_lower = query.lower().strip()

#     # Branch 1: Query for products expiring in exactly X days
#     match = re.search(r"expire in (\d+)\s*day[s]?", query_lower)
#     if match:
#         days = int(match.group(1))
#         count = 0
#         expiring_products = []
#         for product in products:
#             expiry_date = datetime.strptime(product["expiry_date"], "%Y-%m-%d")
#             days_until_expiry = (expiry_date - current_date).days
#             # Only include products that expire exactly in the specified number of days
#             if days_until_expiry == days:
#                 count += product["quantity"]
#                 expiring_products.append(
#                     f"{product['product_name']} (Quantity: {product['quantity']}, Expires in: {days_until_expiry} day{'s' if days_until_expiry != 1 else ''})"
#                 )
#         if expiring_products:
#             product_list_str = "\n".join(f"- {item}" for item in expiring_products)
#             return f"There are {count} units of products that will expire in exactly {days} day{'s' if days != 1 else ''}. Here is the list:\n{product_list_str}"
#         else:
#             return f"No products will expire in exactly {days} day{'s' if days != 1 else ''}."

#     # Branch 2: Query to list all products with their days until expiry
#     elif "list all" in query_lower and "expire" in query_lower:
#         lines = []
#         for product in products:
#             expiry_date = datetime.strptime(product["expiry_date"], "%Y-%m-%d")
#             days_until_expiry = (expiry_date - current_date).days
#             lines.append(f"{product['product_name']} (Quantity: {product['quantity']}, Expires in: {days_until_expiry} day{'s' if days_until_expiry != 1 else ''})")
#         if lines:
#             return "Products and their expiry info:\n" + "\n".join(lines)
#         else:
#             return "No product data available."

#     # Branch 3: Query for expired products
#     elif "expired" in query_lower:
#         count = 0
#         for product in products:
#             expiry_date = datetime.strptime(product["expiry_date"], "%Y-%m-%d")
#             if expiry_date < current_date:
#                 count += product["quantity"]
#         return f"{count} units of products are already expired."

#     # Branch 4: Query for fresh products
#     elif "fresh" in query_lower:
#         # Define 'fresh' as having more than a threshold (e.g., 5 days) before expiry.
#         threshold_days = 5
#         fresh_products = set()
#         for product in products:
#             expiry_date = datetime.strptime(product["expiry_date"], "%Y-%m-%d")
#             if (expiry_date - current_date).days > threshold_days:
#                 fresh_products.add(product["product_name"])
#         if fresh_products:
#             fresh_list = ", ".join(sorted(fresh_products))
#             return f"Fresh products (with more than {threshold_days} days until expiry): {fresh_list}"
#         else:
#             return "No products meet the 'fresh' criteria."

#     else:
#         return "Query not recognized. Please ask about product expiry or freshness."


# def create_product_inventory_query_tool():
#     product_inventory_query_tool = BaseTool(
#     name="product_inventory_query_tool",
#     description="Tool to answer queries about product expiry and freshness using the product database.",
#     function=product_inventory_query,
#     parameters={
#         "query": {
#             "type": "string",
#             "description": "A query about product expiry or freshness, e.g., 'How many products will expire in 7 days?'"
#         }
#     },
#     required=["query"]
# )
#     return product_inventory_query_tool

# def list_all_products_with_expiry() -> str:
#     """
#     Reads the product database from a CSV file and returns a formatted list of all products with their expiry dates.

#     Returns:
#         str: A formatted string listing each product with its product ID, name, buying date, expiry date, and quantity.
#     """
#     products = []
#     try:
#         with open("product_db.csv", newline="") as csvfile:
#             reader = csv.DictReader(csvfile)
#             for row in reader:
#                 row["quantity"] = int(row["quantity"])
#                 products.append(row)
#     except FileNotFoundError:
#         return "Product database not found. Please ensure 'product_db.csv' exists in the working directory."

#     if not products:
#         return "No product data available."

#     lines = []
#     for product in products:
#         line = (
#             f"Product ID: {product['product_id']}, "
#             f"Name: {product['product_name']}, "
#             f"Buying Date: {product['buying_date']}, "
#             f"Expiry Date: {product['expiry_date']}, "
#             f"Quantity: {product['quantity']}"
#         )
#         lines.append(line)

#     return "Product List with Expiry Dates:\n" + "\n".join(lines)


# def create_list_all_products_with_expiry_tool():
#     list_products_tool = BaseTool(
#     name="list_products_with_expiry_tool",
#     description="Lists all products with their expiry dates and details from the product database.",
#     function=list_all_products_with_expiry,
#     parameters={},  # No parameters required
#     required=[]
# )

# # Then register this tool with your tool registry:
#     return list_products_tool




# def get_product_details(product_query: str) -> str:
#     """
#     Reads the product database from a CSV file and returns details for the product(s)
#     that match the provided product_query (case-insensitive).

#     Args:
#         product_query (str): The name (or partial name) of the product to search for.

#     Returns:
#         str: A formatted string containing the details of matching products, or a message if none found.
#     """
#     products = []
#     try:
#         with open("product_db.csv", newline="") as csvfile:
#             reader = csv.DictReader(csvfile)
#             for row in reader:
#                 row["quantity"] = int(row["quantity"])
#                 # If the product name contains the search query (case-insensitive)
#                 if product_query.lower() in row["product_name"].lower():
#                     products.append(row)
#     except FileNotFoundError:
#         return "Product database not found. Please ensure 'product_db.csv' exists in the working directory."

#     if not products:
#         return f"No details found for product '{product_query}'."
#     else:
#         details_lines = []
#         for product in products:
#             details_lines.append(
#                 f"Product ID: {product['product_id']}, "
#                 f"Name: {product['product_name']}, "
#                 f"Buying Date: {product['buying_date']}, "
#                 f"Expiry Date: {product['expiry_date']}, "
#                 f"Quantity: {product['quantity']}"
#             )
#         return "\n".join(details_lines)


def load_products(filename="product_db.csv"):
    """
    Load product data from CSV file.

    Args:
        filename (str): The path to the CSV file.

    Returns:
        list: A list of product dictionaries.
    """
    products = []
    try:
        with open(filename, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                row["quantity"] = int(row["quantity"])
                products.append(row)
    except FileNotFoundError:
        raise FileNotFoundError("Product database not found. Please ensure 'product_db.csv' exists.")

    return products


# Get expired products
def get_expired_products(query: str, current_date=None) -> str:
    if current_date is None:
        current_date = datetime.today()

    products = load_products()
    expired = [
        product for product in products
        if datetime.strptime(product["expiry_date"], "%Y-%m-%d") < current_date
    ]
    return "\n".join(f"{p['product_name']} (Quantity: {p['quantity']}, Expiry: {p['expiry_date']})" for p in expired) or "No expired products."

def create_expired_products_tool():
    return BaseTool(
        name="expired_products_tool",
        description="Returns products that have expired.",
        function=get_expired_products,
        parameters={"query": {"type": "string", "description": "User's query to identify expired products."}},
    )


# Get products expiring within a threshold
def get_products_expiring_within(days: int, current_date=None) -> str:
    if current_date is None:
        current_date = datetime.today()

    products = load_products()
    threshold_date = current_date + timedelta(days=days)
    expiring = [
        product for product in products
        if current_date <= datetime.strptime(product["expiry_date"], "%Y-%m-%d") <= threshold_date
    ]
    return "\n".join(f"{p['product_name']} (Quantity: {p['quantity']}, Expiry: {p['expiry_date']})" for p in expiring) or f"No products expiring within {days} days."

def create_products_expiring_within_tool():
    return BaseTool(
        name="products_expiring_within_tool",
        description="Returns products expiring within a given threshold of days.",
        function=get_products_expiring_within,
        parameters={"days": {"type": "integer", "description": "Threshold in days."}},
        required=["days"]
    )


# Get product details by name
def get_product_details(query: str, product_name: str) -> str:
    products = load_products()
    details = [
        product for product in products
        if product_name.lower() in product["product_name"].lower()
    ]
    return "\n".join(
        f"Product ID: {p['product_id']}, Name: {p['product_name']}, Quantity: {p['quantity']}, Buying Date: {p['buying_date']}, Expiry: {p['expiry_date']}"
        for p in details
    ) or f"No details found for product '{product_name}'."

def create_product_details_tool():
    return BaseTool(
        name="product_details_tool",
        description="Returns details of a specific product.",
        function=get_product_details,
        parameters={"query": {"type": "string", "description": "The user's query."}, "product_name": {"type": "string", "description": "Name of the product."}},
        required=["query", "product_name"]
    )


# Get fresh products
def get_fresh_products(query: str, threshold_days=2, current_date=None) -> str:
    if current_date is None:
        current_date = datetime.today()

    products = load_products()
    fresh = [
        product for product in products
        if (datetime.strptime(product["expiry_date"], "%Y-%m-%d") - current_date).days > threshold_days
    ]
    return "\n".join(f"{p['product_name']} (Quantity: {p['quantity']}, Expiry: {p['expiry_date']})" for p in fresh) or "No fresh products."

def create_fresh_products_tool():
    return BaseTool(
        name="fresh_products_tool",
        description="Returns fresh products with a specified freshness threshold.",
        function=get_fresh_products,
        parameters={"query": {"type": "string", "description": "The user's query."}, "threshold_days": {"type": "integer", "description": "Number of days to consider as fresh."}},
        required=["query"]
    )


def create_inventory_agent(tool_registry):
    agent_config = AzureOpenAIAgentConfig(
        agent_name="chat_agent",
        description="An interactive chat agent",
        model_name="gpt-4o",
        agent_type="ChatAgent",
        tool_registry=tool_registry,
        system_prompt="""
            ### System Prompt for Inventory Agent
                **Role and Objective:**
                You are an intelligent Inventory Management Agent specializing in managing and tracking grocery products within a home. Your goal is to provide accurate information on product expiration, freshness, and availability.

                ### Tools Available:
                1. **Expired Products Tool (`expired_products_tool`):**
                - Identifies products that have already expired.
                - Input: User's query.
                - Output: List of expired products with quantity and expiration date.

                2. **Products Expiring Within Tool (`products_expiring_within_tool`):**
                - Returns products that will expire within a specified number of days.
                - Input: User's query and threshold in days.
                - Output: List of products expiring within the threshold.

                3. **Product Details Tool (`product_details_tool`):**
                - Retrieves detailed information about a specific product by name.
                - Input: User's query and product name.
                - Output: Product ID, name, quantity, buying date, and expiration date.

                4. **Fresh Products Tool (`fresh_products_tool`):**
                - Lists products with sufficient freshness based on a threshold in days.
                - Input: User's query and freshness threshold.
                - Output: List of fresh products with quantity and expiration date.

                ### Guidelines:
                - Always store the user's message and retrieve conversation context before generating responses.
                - Use the appropriate tool based on the user's query.
                - If the user asks about expired products, call the `expired_products_tool`.
                - For upcoming expiration inquiries, use `products_expiring_within_tool`.
                - To provide product-specific details, use the `product_details_tool`.
                - To show fresh products, utilize `fresh_products_tool`.

                Your goal is to assist users in efficiently managing their grocery inventory while minimizing waste and optimizing usage.
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
        api_base="https://aoi-iiit-hack-2.openai.azure.com/" ,  # Use default OpenAI API base
        api_version=os.getenv("AZURE_OPENAI_API_VERSION") or "2024-12-01-preview",
        organization=None  # Use default organization
    )

    # Create Azure OpenAI agent with memory capabilities
    onboarding_agent = AzureOpenAIAgent(
        config=onboarding_agent_config
    )

    return onboarding_agent


def create_alternate_meal_agent(tool_registry):
    agent_config = AzureOpenAIAgentConfig(
        agent_name="alternate_meal_agent",
        description="An intelligent agent for suggesting alternate meals based on pantry inventory.",
        model_name="gpt-4o",
        agent_type="ChatAgent",
        tool_registry=tool_registry,
        system_prompt="""
            ### System Prompt for Alternate Meal Agent
                **Role and Objective:**
                You are an intelligent Meal Planning Agent that suggests alternate meals based on the user's pantry inventory.

                ### Tools Available:

                1. **Products Expiring Within Tool (`products_expiring_within_tool`):**
                - Returns products that will expire within a specified number of days.
                - Input: User's query and threshold in days.
                - Output: List of products expiring within the threshold.

                2. **Fresh Products Tool (`fresh_products_tool`):**
                - Lists products with sufficient freshness based on a threshold in days.
                - Input: User's query and freshness threshold.
                - Output: List of fresh products with quantity and expiration date.

                3. **Generate Recipe Tool (`generate_recipe_tool`):**
                - Generates a recipe based on the dish suggested.
                - Input: Dish name.
                - Output: Step-by-step recipe instructions.

                ### Guidelines:
                - When the user asks for a meal suggestion, inquire if they want to use fresh products or those expiring within 5 days.
                - Formulate a dish based on the selected option.
                - If the user requests a recipe, use the `generate_recipe_tool` to provide a recipe for the suggested dish.
                - Store the user's message and retrieve conversation context before generating responses.

                Your goal is to help users make optimal use of pantry items and minimize food waste.
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


def create_classifier_agent(tool_registry):
    """Create a classifier agent for language and task detection."""
    config = AzureOpenAIAgentConfig(
        agent_name="chat_agent",
        description="Language and task classifier for routing messages",
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="gpt-4o",
        agent_type="ChatAgent",
        tool_registry=tool_registry,
        is_streaming=True,
        system_prompt=""""You are a classifier. Your job is to determine the best agent based on the user's message:
        1. If the message requests or implies a need for a meal plan or if the user appears to be a new user, return 'onboarding_agent'
        2. If the message requests information about inventory like what products are expiring or how many products are fresh or any product details, return 'inventory_agent'
        """,
    )

    agent = AzureOpenAIAgent(config)
    return agent



def create_meal_plan_tool():
    meal_plan_tool = BaseTool(
        name="meal_plan_tool",
        description="Returns the meal plan (Breakfast, Lunch, Dinner) for a given day from meal_plan.csv.",
        function=get_meal_plan_for_day,
        parameters={
            "day": {
                "type": "string",
                "description": "The day for which to retrieve the meal plan (e.g., 'Monday')."
            }
        },
        required=["day"]
    )
    return meal_plan_tool


def get_meal_plan_for_day(day: str) -> str:
    """
    Reads the meal_plan.csv file and returns the meal plan for the given day.

    The CSV is expected to have the following columns:
    Day, Breakfast, Lunch, Dinner

    Args:
        day (str): The day for which to retrieve the meal plan (e.g., "Monday").

    Returns:
        str: A formatted string with the meal plan details or an error message if not found.
    """
    try:
        with open("meal_plan.csv", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row["Day"].strip().lower() == day.strip().lower():
                    breakfast = row.get("Breakfast", "N/A")
                    lunch = row.get("Lunch", "N/A")
                    dinner = row.get("Dinner", "N/A")
                    return (f"Meal Plan for {day}:\n"
                            f"Breakfast: {breakfast}\n"
                            f"Lunch: {lunch}\n"
                            f"Dinner: {dinner}")
        return f"No meal plan found for {day}."
    except FileNotFoundError:
        return "Meal plan CSV file not found. Please ensure 'meal_plan.csv' exists in the working directory."



def generate_recipe(food_item: str) -> str:
    """
    Uses the OpenAI API to generate a recipe for the given food item.

    Args:
        food_item (str): The name of the food item.

    Returns:
        str: A detailed recipe including ingredients and steps.
    """
    prompt = (
        f"Provide a detailed recipe for {food_item}. "
        "Include a list of ingredients with quantities and step-by-step instructions for preparation."
    )

    # Make a call to the OpenAI API to generate the recipe
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a knowledgeable culinary assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        recipe = response['choices'][0]['message']['content'].strip()
        return recipe
    except Exception as e:
        return f"An error occurred while generating the recipe: {e}"




def create_generate_recipe_tool():
    recipe_tool = BaseTool(
        name="generate_recipe_tool",
        description="Generates a detailed recipe for a given food item using AI.",
        function=generate_recipe,
        parameters={
            "food_item": {
                "type": "string",
                "description": "The name of the food item to generate a recipe for."
            }
        },
        required=["food_item"]
    )
    return recipe_tool


def create_meal_recipe_agent(tool_registry):
    """
    Creates an agent that, given a day and a meal type (e.g., Breakfast),
    retrieves the corresponding dish from the meal plan and returns a generated recipe.
    """
    agent_config = AzureOpenAIAgentConfig(
        agent_name="meal_recipe_agent",
        description="An agent that returns a dish for a given day and meal type (Breakfast, Lunch, Dinner). and also the recipe of it upon asking",
        model_name="gpt-4o",
        agent_type="ChatAgent",
        tool_registry=tool_registry,
        system_prompt="""
You are a meal recipe agent. When a user provides a day and a meal type (Breakfast, Lunch, Dinner), do the following:
1. Use the 'meal_plan_tool' to retrieve the meal plan for the specified day.
2. Extract the dish corresponding to the given meal type.
3. Use the 'generate_recipe_tool' to generate a detailed recipe for that dish upon asking.
Return the generated recipe to the user.
""",
        api_key=os.getenv("OPENAI_API_KEY"),
        api_base="https://aoi-iiit-hack-2.openai.azure.com/",
        api_version=os.getenv("AZURE_OPENAI_API_VERSION") or "2024-12-01-preview",
        organization=None
    )
    agent = AzureOpenAIAgent(config=agent_config)
    return agent



def update_pantry_and_shopping_list(dish: str, recipe: str, user_inventory_input: str) -> None:
    """
    Extracts ingredients from the recipe, compares them with the user's input,
    updates the pantry (product_db.csv) with items the user already has, and updates the shopping list (shopping_list.csv)
    with missing ingredients.

    For new pantry items, generates a random expiry date (between 5 and 15 days from today) and uses today as the buying date.

    Args:
        dish (str): The name of the dish.
        recipe (str): The generated recipe text (expected to include an "Ingredients:" section).
        user_inventory_input (str): Comma-separated list of ingredients the user already has.
    """
    # Extract ingredients from the recipe.
    ingredients = []
    for line in recipe.splitlines():
        if line.lower().startswith("ingredients:"):
            ing_line = line[len("ingredients:"):].strip()
            ingredients = [item.strip().lower() for item in ing_line.split(",") if item.strip()]
            break

    if not ingredients:
        print(f"Could not extract ingredients for '{dish}'.")
        return

    # Parse the user's pantry input.
    user_inventory = [item.strip().lower() for item in user_inventory_input.split(",") if item.strip()]

    # Load existing pantry from product_db.csv.
    pantry = []
    try:
        with open("product_db.csv", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                row["quantity"] = int(row["quantity"])
                pantry.append(row)
    except FileNotFoundError:
        pantry = []

    # Helper to check if an ingredient exists in the pantry.
    def in_pantry(ingredient):
        for item in pantry:
            if item["product_name"].lower() == ingredient:
                return True
        return False

    # Update pantry: add any ingredients from user's input that are not already in the pantry.
    current_date = datetime.today()
    for ing in user_inventory:
        if not in_pantry(ing):
            new_item = {
                "product_id": str(len(pantry) + 1),  # Simple ID generation.
                "product_name": ing,
                "buying_date": current_date.strftime("%Y-%m-%d"),
                "expiry_date": (current_date + timedelta(days=random.randint(5,15))).strftime("%Y-%m-%d"),
                "quantity": "1"
            }
            pantry.append(new_item)
            print(f"Added '{ing}' to pantry (product_db.csv).")

    # Write the updated pantry back to product_db.csv.
    with open("product_db.csv", "w", newline="") as csvfile:
        fieldnames = ["product_id", "product_name", "buying_date", "expiry_date", "quantity"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in pantry:
            writer.writerow(row)

    # Identify missing ingredients: those in the recipe but not in the user's input.
    missing = [ing for ing in ingredients if ing not in user_inventory]

    # Update shopping list (shopping_list.csv) with missing items.
    if missing:
        print(f"Missing ingredients for '{dish}': {', '.join(missing)}")
        shopping_list = []
        try:
            with open("shopping_list.csv", newline="") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    shopping_list.append(row)
        except FileNotFoundError:
            shopping_list = []

        for item in missing:
            if not any(row["ingredient"].lower() == item for row in shopping_list):
                shopping_list.append({"ingredient": item, "dish": dish})

        with open("shopping_list.csv", "w", newline="") as csvfile:
            fieldnames = ["ingredient", "dish"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in shopping_list:
                writer.writerow(row)
    else:
        print(f"All ingredients for '{dish}' are present in your pantry.")

    # Finally, print the missing ingredients that need to be bought.
    if missing:
        print(f"\nFor '{dish}', you need to buy: {', '.join(missing)}")
    else:
        print(f"Your pantry has all the ingredients for '{dish}'.")





def setup_agent():
    """
    Set up the AzureOpenAI agent with memory capabilities and return the orchestrator and agent.

    Returns:
        tuple: A tuple containing the orchestrator and the agent.
    """
    # Set up memory components
    tool_registry = setup_memory_components()
    tool_registry.register_tool(create_expired_products_tool())
    tool_registry.register_tool(create_products_expiring_within_tool())
    tool_registry.register_tool(create_product_details_tool())
    tool_registry.register_tool(create_fresh_products_tool())
    tool_registry.register_tool(import_to_csv_tool())
    tool_registry.register_tool(create_generate_recipe_tool())
    tool_registry.register_tool(create_meal_plan_tool())


    # Set up registry and orchestrator
    agent_registry = AgentRegistry()
    inventory_agent = create_inventory_agent(tool_registry)
    onboarding_agent = create_onboarding_agent(tool_registry)
    meal_recipe_agent = create_meal_recipe_agent(tool_registry)
    agent_registry.register_agent(inventory_agent)
    agent_registry.register_agent(onboarding_agent)
    agent_registry.register_agent(meal_recipe_agent)
    agent_registry.register_agent(create_alternate_meal_agent(tool_registry))


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


