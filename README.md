# AI-Powered Grocery Manager

## Overview
The AI-Powered Grocery Manager is a multi-agent system designed to help users efficiently manage their groceries and meal plans. It aims to minimize food waste and impulse buying by suggesting dishes based on available ingredients and tracking inventory status.

## Features
- **Onboarding Process**: Asks users questions to create a personalized meal plan.
- **Meal Plan Management**: Recommends dishes from the meal plan and provides recipes.
- **Inventory-Based Suggestions**: Suggests new dishes based on available groceries.
- **Inventory Tracking**: Provides alerts for items nearing expiration and low stock.
- **Shopping Assistance**: Generates a shopping list based on stock duration and compares market prices for the best deals.
- **Inventory Updates**: Updates the CSV files upon cooking and adding new items.

## File Structure
```
.env
.gitignore
grocery_db.csv
meal_plan.csv
multi-agent.py
product_db.csv
README.md
```

- **grocery_db.csv**: Tracks available groceries and their quantities.
- **meal_plan.csv**: Stores the meal plan based on user preferences.
- **multi-agent.py**: The main Python script handling all functionalities.
- **product_db.csv**: Contains data from different stores and their prices.
- **README.md**: Documentation for the project.

## How It Works
1. **User Onboarding**: Collects user preferences to generate a meal plan.
2. **Dish Recommendation**: Suggests dishes from the meal plan or based on available ingredients.
3. **Inventory Management**: Monitors expiration dates and low stock levels.
4. **Smart Shopping**: Creates a shopping list and compares market prices.
5. **CSV Updates**: Updates the inventory after cooking or restocking.

## Setup Instructions
1. Clone the repository.
2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
3. Activate the virtual environment:
   - On Windows:
     ```bash
     .\venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
4. Run the main script:
   ```bash
   python multi-agent.py
   ```

## Future Enhancements
- Integration with external APIs for real-time price comparisons.
- Enhanced AI models for meal planning and recommendation.
- Mobile app interface for better user experience.

## License
This project is licensed under the MIT License.

