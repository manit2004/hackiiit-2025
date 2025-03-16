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
def get_expired_products(current_date=None) -> str:
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
        required=["query"]
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

def get_low_stock_products(threshold=5, filename="product_db.csv"):
    """
    Identifies products that are running low in stock.

    Args:
        threshold (int): The quantity threshold below which products are considered low stock.
        filename (str): The CSV file containing product data.

    Returns:
        str: A formatted string listing products with quantities below the threshold.
    """
    try:
        with open(filename, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            low_stock = [
                f"{row['product_name']} (Quantity: {row['quantity']})"
                for row in reader if int(row['quantity']) < threshold
            ]
    except FileNotFoundError:
        return "Product database not found. Please ensure 'product_db.csv' exists."

    return "\n".join(low_stock) or "No products are running low."


def update_product_quantity(product_name: str, new_quantity: int) -> str:
    """
    Updates the quantity of all products whose name matches the given product_name
    in the product_db.csv file to the new_quantity.

    Args:
        product_name (str): The product name (or partial name) to search for.
        new_quantity (int): The new quantity to set.

    Returns:
        str: A message indicating the update result.
    """
    products = load_products()
    if not products:
        return "No products loaded from the database."

    updated = False
    for product in products:
        if product_name.lower() in product["product_name"].lower():
            product["quantity"] = new_quantity
            updated = True

    if not updated:
        return f"No product found matching '{product_name}'."

    # Write the updated products list back to the CSV file.
    fieldnames = ["product_id", "product_name", "buying_date", "expiry_date", "quantity"]
    try:
        with open("product_db.csv", "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for product in products:
                writer.writerow(product)
        return f"Quantity for product(s) matching '{product_name}' updated to {new_quantity}."
    except Exception as e:
        return f"An error occurred while updating the database: {e}"


def create_update_product_quantity_tool():
    update_tool = BaseTool(
        name="update_product_quantity_tool",
        description="Updates the quantity of a product (or products) in product_db.csv that match the given product name to the specified new quantity.",
        function=update_product_quantity,
        parameters={
            "product_name": {
                "type": "string",
                "description": "The name (or partial name) of the product to update."
            },
            "new_quantity": {
                "type": "integer",
                "description": "The new quantity to set for the product."
            }
        },
        required=["product_name", "new_quantity"]
    )
    return update_tool


def create_low_stock_products_tool():
    return BaseTool(
        name="low_stock_products_tool",
        description="Identifies products that are running low in stock.",
        function=get_low_stock_products,
        parameters={"threshold": {"type": "integer", "description": "Quantity threshold for low stock."}},
        required=["threshold"]
    )

def create_inventory_agent(tool_registry):
    """
    Creates an inventory management agent that:
      - Tracks product expiration and freshness.
      - Provides product details.
      - Identifies products that are running low in stock.
    """
    agent_config = AzureOpenAIAgentConfig(
        agent_name="inventory_agent",
        description="An interactive inventory management agent for tracking product status and freshness.",
        model_name="gpt-4o",
        agent_type="ChatAgent",
        tool_registry=tool_registry,
        system_prompt="""
### System Prompt for Inventory Agent

**Role and Objective:**
You are an intelligent Inventory Management Agent specializing in managing and tracking grocery products within a home. Your goal is to provide accurate information on product expiration, freshness, availability, and low-stock alerts.

**Tools Available:**
1. **Expired Products Tool (`expired_products_tool`):**
   - Identifies products that have already expired.
   - **Input:** User's query.
   - **Output:** List of expired products with quantity and expiration date.

2. **Products Expiring Within Tool (`products_expiring_within_tool`):**
   - Returns products that will expire within a specified number of days.
   - **Input:** User's query and threshold in days.
   - **Output:** List of products expiring within the threshold.

3. **Product Details Tool (`product_details_tool`):**
   - Retrieves detailed information about a specific product by name.
   - **Input:** User's query and product name.
   - **Output:** Product ID, name, quantity, buying date, and expiration date.

4. **Fresh Products Tool (`fresh_products_tool`):**
   - Lists products with sufficient freshness based on a threshold in days.
   - **Input:** User's query and freshness threshold.
   - **Output:** List of fresh products with quantity and expiration date.

5. **Low Stock Products Tool (`low_stock_products_tool`):**
   - Identifies products that are running low in stock.
   - **Input:** Quantity threshold for low stock.
   - **Output:** A list of products with quantities below the threshold.

**Guidelines:**
- Store the user's message and retrieve the conversation context before generating responses.
- Use the appropriate tool based on the user's query.
- For expired products, use `expired_products_tool`.
- For upcoming expiration inquiries, use `products_expiring_within_tool`.
- For product-specific details, use `product_details_tool`.
- To show fresh products, use `fresh_products_tool`.
- For low stock alerts, use `low_stock_products_tool`.

Your goal is to assist users in efficiently managing their grocery inventory while minimizing waste and optimizing usage.
        """,
        api_key=os.getenv("OPENAI_API_KEY"),
        api_base="https://aoi-iiit-hack-2.openai.azure.com/",
        api_version=os.getenv("AZURE_OPENAI_API_VERSION") or "2024-12-01-preview",
        organization=None
    )
    agent = AzureOpenAIAgent(config=agent_config)
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

def get_ingredients(food_item: str) -> str:
    """
    Uses the OpenAI API to get only the ingredients required for the given food item.

    Args:
        food_item (str): The name of the food item.

    Returns:
        str: A list of ingredients with quantities.
    """
    prompt = (
        f"List the ingredients and their quantities required to make {food_item}. "
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a knowledgeable culinary assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        ingredients = response['choices'][0]['message']['content'].strip()
        return ingredients
    except Exception as e:
        return f"An error occurred while fetching the ingredients: {e}"


def create_get_ingredients_tool():
    return BaseTool(
        name="get_ingredients_tool",
        description="Fetches a list of ingredients required for a given food item.",
        function=get_ingredients,
        parameters={"food_item": {"type": "string", "description": "The name of the food item."}},
        required=["food_item"]
    )


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


def create_meal_plan_range_tool():
    """
    Creates a tool that retrieves meal plans for a range of days starting from today.
    """
    meal_plan_range_tool = BaseTool(
        name="meal_plan_range_tool",
        description="Returns meal plans (Breakfast, Lunch, Snack, Dinner) for a given number of days starting from today.",
        function=get_meal_plans_for_range,
        parameters={
            "num_days": {
                "type": "integer",
                "description": "The number of days from today for which to retrieve the meal plans."
            }
        },
        required=["num_days"]
    )
    return meal_plan_range_tool



def get_meal_plans_for_range(num_days: int) -> str:
    """
    Retrieves meal plans from the current day for the next 'num_days' days.
    The CSV file 'meal_plan.csv' is expected to have columns: Day, Breakfast, Lunch, Snack, Dinner.

    Args:
        num_days (int): The number of days (starting from today) for which to retrieve the meal plans.

    Returns:
        str: A formatted string of the meal plans for each day.
    """
    # Get current day name (e.g., "Monday")
    today = datetime.today()
    day_names = []
    for i in range(num_days):
        day = (today + timedelta(days=i)).strftime("%A")
        day_names.append(day)

    # Build output by reading meal_plan.csv for each day in the range.
    meal_plans = []
    try:
        with open("meal_plan.csv", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            # Create a dictionary mapping day name (lowercase) to its row.
            meal_dict = { row["Day"].strip().lower(): row for row in reader }
    except FileNotFoundError:
        return "Meal plan CSV file not found. Please ensure 'meal_plan.csv' exists in the working directory."

    for day in day_names:
        row = meal_dict.get(day.lower())
        if row:
            # Support optional "Snack" column if exists, otherwise use "N/A"
            snack = row.get("Snack", "N/A")
            plan_str = (f"Meal Plan for {day}:\n"
                        f"  Breakfast: {row.get('Breakfast', 'N/A')}\n"
                        f"  Lunch: {row.get('Lunch', 'N/A')}\n"
                        f"  Snack: {snack}\n"
                        f"  Dinner: {row.get('Dinner', 'N/A')}\n")
        else:
            plan_str = f"No meal plan found for {day}."
        meal_plans.append(plan_str)

    return "\n".join(meal_plans)



def create_ingredients_for_meal_plan_tool():
    ingredients_tool = BaseTool(
        name="ingredients_for_meal_plan_tool",
        description="Extracts and consolidates ingredients for all meals from a given meal plan string.",
        function=get_ingredients_for_meal_plan,
        parameters={
            "plan_str": {
                "type": "string",
                "description": "The meal plan string (e.g., the output from get_meal_plans_for_range) from which to extract ingredients."
            }
        },
        required=["plan_str"]
    )
    return ingredients_tool



def get_ingredients_for_meal_plan(plan_str: str) -> str:
    """
    Given a meal plan string containing details for multiple days and meals,
    extracts each dish and uses the generate_recipe function to generate its recipe.
    Then, it extracts the ingredients (from the "Ingredients:" line) for each dish
    and returns a consolidated, formatted list of ingredients for all meals.

    Args:
        plan_str (str): The meal plan string (e.g., the output from get_meal_plans_for_range).

    Returns:
        str: A formatted string listing each dish with its corresponding ingredients.
    """
    # Split the meal plan into lines.
    lines = plan_str.splitlines()
    # Dictionary to hold dish names keyed by meal type (e.g., Breakfast, Lunch, Dinner, Snack)
    dishes = {}
    for line in lines:
        if ":" in line:
            label, value = line.split(":", 1)
            label = label.strip().lower()
            dish = value.strip()
            # Consider common meal labels (adjust if needed)
            if label in ["breakfast", "lunch", "dinner", "snack"]:
                dishes[label] = dish

    consolidated = {}
    # For each dish, generate a recipe and extract ingredients.
    for meal, dish in dishes.items():
        print(f"Processing {meal} dish: {dish}")
        recipe = generate_recipe(dish)
        ingredients = None
        # Look for a line starting with "Ingredients:" (case-insensitive)
        for r_line in recipe.splitlines():
            if r_line.lower().startswith("ingredients:"):
                ing_line = r_line[len("ingredients:"):].strip()
                # Split by commas; adjust separator if needed
                ingredients = [item.strip() for item in ing_line.split(",") if item.strip()]
                break
        if ingredients:
            consolidated[dish] = ingredients
        else:
            consolidated[dish] = ["No ingredients found."]

    # Format the output.
    output_lines = []
    for dish, ingredients in consolidated.items():
        output_lines.append(f"{dish}: {', '.join(ingredients)}")

    return "\n".join(output_lines)



def check_missing_ingredients(ingredients_str: str) -> str:
    """
    Given a comma-separated string of ingredients, this function checks the product_db.csv
    to determine which ingredients are not present in the pantry. It returns a formatted string
    listing all missing ingredients.

    Args:
        ingredients_str (str): A comma-separated string of ingredients.

    Returns:
        str: A message with the list of missing ingredients or a message indicating that all ingredients are present.
    """
    # Parse the input into a list of ingredients (in lowercase for comparison)
    user_ingredients = [ing.strip().lower() for ing in ingredients_str.split(",") if ing.strip()]

    # Load existing product names from product_db.csv
    products = []
    try:
        with open("product_db.csv", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                product_name = row["product_name"].strip().lower()
                products.append(product_name)
    except FileNotFoundError:
        return "Product database not found. Please ensure 'product_db.csv' exists in the working directory."

    # Determine missing ingredients (ingredients not found in products)
    missing = [ing for ing in user_ingredients if ing not in products]

    if missing:
        return "Missing ingredients: " + ", ".join(missing)
    else:
        return "All ingredients are present in the product database."





def create_missing_ingredients_tool():
    missing_tool = BaseTool(
        name="missing_ingredients_tool",
        description="Checks which ingredients from the provided list are not present in the product_db.csv.",
        function=check_missing_ingredients,
        parameters={
            "ingredients_str": {
                "type": "string",
                "description": "A comma-separated string of ingredients to check against the pantry."
            }
        },
        required=["ingredients_str"]
    )
    return missing_tool


def find_cheapest_option_for_ingredient(ingredient: str) -> str:
    """
    Searches the grocery_db.csv for a given ingredient, returns all available options from different vendors,
    and then determines and returns the cheapest option.

    Args:
        ingredient (str): The name of the ingredient to search for.

    Returns:
        str: A formatted string listing all vendor options with prices and the cheapest option.
    """
    matches = []
    try:
        with open("grocery_db.csv", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Check if the product name contains the ingredient (case-insensitive)
                if ingredient.lower() in row["grocery_item"].lower():
                    try:
                        price = float(row["price"])
                    except ValueError:
                        continue
                    matches.append({
                        "grocery_item": row["grocery_item"],
                        "price": price,
                        "stock": row["stock"],
                        "vendor": row["vendor"]
                    })
    except FileNotFoundError:
        return "Grocery database not found. Please ensure 'grocery_db.csv' exists in the working directory."

    if not matches:
        return f"No options found for ingredient '{ingredient}'."

    # Sort matches by price (lowest first)
    matches_sorted = sorted(matches, key=lambda x: x["price"])
    cheapest = matches_sorted[0]

    # Build output string with all options
    options_str = "\n".join(
        f"- {item['grocery_item']} from {item['vendor']} at ${item['price']} (Stock: {item['stock']})"
        for item in matches_sorted
    )

    output = (
        f"Options for '{ingredient}':\n{options_str}\n\n"
        f"Cheapest option: {cheapest['grocery_item']} from {cheapest['vendor']} at ${cheapest['price']} (Stock: {cheapest['stock']})."
    )
    return output



def create_cheapest_option_tool():
    cheapest_tool = BaseTool(
        name="cheapest_option_tool",
        description="Finds all vendor options for a given ingredient from grocery_db.csv and returns the cheapest one.",
        function=find_cheapest_option_for_ingredient,
        parameters={
            "ingredient": {
                "type": "string",
                "description": "The name of the ingredient to search for."
            }
        },
        required=["ingredient"]
    )
    return cheapest_tool


def create_shopping_assistant(tool_registry):
    """
    Creates a shopping assistant agent that:
      - Retrieves the meal plan for a given number of days starting from today.
      - Extracts all ingredients from the meal plan.
      - Checks which ingredients are missing in the pantry (product_db.csv).
      - For each missing ingredient, finds vendor options and returns the best price.
      - Consolidates the missing items into a shopping cart list.
      - When the user confirms they have purchased the missing items, uses the add_product_tool
        to add these items (with the current date as buying date and a random expiry date) into product_db.csv.
    """
    agent_config = AzureOpenAIAgentConfig(
        agent_name="shopping_assistant",
        description="An intelligent shopping assistant for meal planning and optimized grocery purchasing.",
        model_name="gpt-4o",
        agent_type="ChatAgent",
        tool_registry=tool_registry,
        system_prompt="""
**Role and Objective:**
You are an intelligent Shopping Assistant dedicated to helping users manage their meal planning and grocery shopping efficiently. Your tasks include:
1. Retrieving the meal plan for a specified number of days starting from today.
2. Extracting and consolidating all ingredients from these meal plans.
3. Comparing the extracted ingredients with the user's pantry (recorded in product_db.csv) to identify missing ingredients.
4. For each missing ingredient, searching the marketplace (grocery_db.csv) to find vendor options and determine the cheapest option.
5. Compiling a comprehensive shopping cart list with the missing items, best prices, and vendor information.
6. When the user indicates that they have purchased certain missing items, call the add_product_tool to add those items to product_db.csv, using the current date as the buying date and a randomly generated expiry date.

**Tools Available:**
1. **Meal Plan Range Tool (`meal_plan_range_tool`):**
   - Retrieves meal plans (Breakfast, Lunch, Snack, Dinner) for a given number of days starting from today.
   - **Input:** Number of days.
   - **Output:** A formatted string containing the meal plans for each day.

2. **Ingredients for Meal Plan Tool (`ingredients_for_meal_plan_tool`):**
   - Extracts and consolidates ingredients from a provided meal plan string.
   - **Input:** Meal plan string (e.g., output from `meal_plan_range_tool`).
   - **Output:** A formatted list of ingredients across all meals.

3. **Missing Ingredients Tool (`missing_ingredients_tool`):**
   - Checks which ingredients from a given list are not present in the pantry (product_db.csv).
   - **Input:** Comma-separated list of ingredients.
   - **Output:** A list of missing ingredients.

4. **Cheapest Option Tool (`cheapest_option_tool`):**
   - Searches the marketplace (grocery_db.csv) for a specified ingredient and returns all vendor options along with their prices, highlighting the cheapest option.
   - **Input:** Ingredient name.
   - **Output:** A list of vendor options with prices, with the cheapest option clearly indicated.

5. **Add Product Tool (`add_product_tool`):**
   - Adds a new product to product_db.csv using the current date as the buying date and a random expiry date.
   - **Input:** Product name and quantity.
   - **Output:** Confirmation that the product has been added.

**Guidelines:**
- When assisting the user:
  1. Use the `meal_plan_range_tool` to get the meal plans for the requested number of days.
  2. Extract the ingredients using the `ingredients_for_meal_plan_tool`.
  3. Determine missing ingredients using the `missing_ingredients_tool`.
  4. Find the cheapest option for each missing ingredient with the `cheapest_option_tool`.
  5. Compile a shopping cart with all missing ingredients, their best prices, and vendor information.
  6. If the user confirms that they have purchased some or all missing items, call the `add_product_tool` for each purchased item to update the pantry database (product_db.csv).

- Always begin by storing the user's message and retrieving the conversation context before generating your final response.
- Provide clear, actionable, and concise output to guide the user in their grocery purchasing decisions.

Your ultimate goal is to help the user minimize food waste and save money by ensuring they purchase only the necessary items at the best available prices.
        """,
        api_key=os.getenv("OPENAI_API_KEY"),
        api_base="https://aoi-iiit-hack-2.openai.azure.com/",
        api_version=os.getenv("AZURE_OPENAI_API_VERSION") or "2024-12-01-preview",
        organization=None
    )
    agent = AzureOpenAIAgent(config=agent_config)
    return agent



def create_cook_agent(tool_registry):
    agent_config = AzureOpenAIAgentConfig(
        agent_name="cook_agent",
        description="An agent that estimates ingredient usage and updates product quantities after cooking.",
        model_name="gpt-4o",
        agent_type="ChatAgent",
        tool_registry=tool_registry,
        system_prompt="""
**Role and Objective:**
You are an intelligent Cook Assistant Agent that helps users track ingredient usage and update inventory after cooking.

**Steps to Follow:**
1. Use the `get_ingredients_tool` to fetch the list of ingredients and their quantities for the dish cooked.
2. For each ingredient, use the `product_details_tool` to check the current quantity in the inventory.
3. Estimate the quantity used and calculate the updated quantity.
4. Use the `update_product_quantity_tool` to update the quantity in the database.

Your goal is to accurately track consumption and maintain an updated inventory.
        """,
        api_key=os.getenv("OPENAI_API_KEY"),
        api_base="https://aoi-iiit-hack-2.openai.azure.com/",
        api_version=os.getenv("AZURE_OPENAI_API_VERSION") or "2024-12-01-preview",
        organization=None
    )
    return AzureOpenAIAgent(config=agent_config)


def create_add_product_tool():
    add_product_tool = BaseTool(
        name="add_product_tool",
        description="Adds a new product to product_db.csv with the current date as buying date, a random expiry date, and the specified quantity.",
        function=add_product_to_db,
        parameters={
            "product_name": {
                "type": "string",
                "description": "The name of the product to add."
            },
            "quantity": {
                "type": "integer",
                "description": "The quantity to set for the product."
            }
        },
        required=["product_name", "quantity"]
    )
    return add_product_tool


def add_product_to_db(product_name: str, quantity: int) -> str:
    """
    Adds a new product to product_db.csv with the given product name and quantity.
    Uses the current date as the buying date and generates a random expiry date (between 5 and 15 days from today).
    The CSV format is: product_id, product_name, buying_date, expiry_date, quantity.
    
    Args:
        product_name (str): The name of the product to add.
        quantity (int): The quantity to set for the product.
        
    Returns:
        str: A message indicating success or failure.
    """
    from datetime import datetime, timedelta
    import random
    import csv

    # Define file name
    filename = "product_db.csv"
    
    # Get current date and format it
    today = datetime.today()
    buying_date = today.strftime("%Y-%m-%d")
    
    # Generate random expiry date: between 5 and 15 days from today
    expiry_date = (today + timedelta(days=random.randint(5, 15))).strftime("%Y-%m-%d")
    
    # Load existing products to determine the next product_id
    products = []
    max_id = 0
    try:
        with open(filename, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                products.append(row)
                try:
                    pid = int(row["product_id"])
                    if pid > max_id:
                        max_id = pid
                except (ValueError, KeyError):
                    continue
    except FileNotFoundError:
        # If file doesn't exist, we'll create a new one
        pass

    new_product_id = max_id + 1
    
    # Create the new product entry
    new_product = {
        "product_id": new_product_id,
        "product_name": product_name,
        "buying_date": buying_date,
        "expiry_date": expiry_date,
        "quantity": quantity
    }
    products.append(new_product)
    
    # Write all products back to the CSV file
    fieldnames = ["product_id", "product_name", "buying_date", "expiry_date", "quantity"]
    try:
        with open(filename, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for prod in products:
                writer.writerow(prod)
        return f"Product '{product_name}' added successfully with quantity {quantity} (Buying Date: {buying_date}, Expiry Date: {expiry_date})."
    except Exception as e:
        return f"An error occurred while updating the database: {e}"


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
    tool_registry.register_tool(create_meal_plan_range_tool())
    tool_registry.register_tool(create_ingredients_for_meal_plan_tool())
    tool_registry.register_tool(create_missing_ingredients_tool())
    tool_registry.register_tool(create_cheapest_option_tool())
    tool_registry.register_tool(create_low_stock_products_tool())
    tool_registry.register_tool(create_update_product_quantity_tool())
    tool_registry.register_tool(create_add_product_tool())



    # Set up registry and orchestrator
    agent_registry = AgentRegistry()
    inventory_agent = create_inventory_agent(tool_registry)
    onboarding_agent = create_onboarding_agent(tool_registry)
    meal_recipe_agent = create_meal_recipe_agent(tool_registry)
    shopping_assistant_agent = create_shopping_assistant(tool_registry)
    cook_agent = create_cook_agent(tool_registry)

    agent_registry.register_agent(inventory_agent)
    agent_registry.register_agent(onboarding_agent)
    agent_registry.register_agent(meal_recipe_agent)
    agent_registry.register_agent(cook_agent)
    agent_registry.register_agent(create_alternate_meal_agent(tool_registry))
    agent_registry.register_agent(shopping_assistant_agent)


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


