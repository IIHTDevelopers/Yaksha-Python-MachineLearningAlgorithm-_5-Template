import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib


# ------------------------ #
#  Function 1: Load & Prep
# ------------------------ #
def load_and_prepare_data(path="car_resale.csv"):
    """
    Load and prepare the car resale dataset.
    
    TODO:
    1. Load the CSV file from the given path
    2. Fill missing values in 'fuel_type' and 'brand_score'
    3. Encode categorical features
    4. Print a success message
    5. Return the processed DataFrame
    """
    # Your code here
    pass


# ------------------- #
# Function 2: EDA
# ------------------- #
def explore_data(df):
    """
    Explore the dataset and print key statistics.
    
    TODO:
    1. Calculate the maximum resale price
    2. Calculate the average resale price
    3. Print these statistics with appropriate formatting
    """
    # Your code here
    pass


# ----------------------------------- #
# Function 3: Prediction Demo (Linear)
# ----------------------------------- #
def prediction_demo(model, X_sample):
    """
    Demonstrate a prediction using the trained model.
    
    TODO:
    1. Use the model to predict the resale price for the given sample
    2. Print the prediction with appropriate formatting
    """
    # Your code here
    pass


# ----------------------------- #
# Function 4: Custom Cost (MSE)
# ----------------------------- #
def cost_function(y_true, y_pred):
    """
    Calculate the Mean Squared Error between true and predicted values.
    
    TODO:
    1. Implement the MSE calculation
    2. Return the calculated MSE
    """
    # Your code here
    pass


# ----------------------------- #
# Function 5: Train & Evaluate
# ----------------------------- #
def train_and_evaluate(X_train, y_train, X_test, y_test, path="car_resale_model.pkl"):
    """
    Train a linear regression model, save it, and evaluate its performance.
    
    TODO:
    1. Create and train a LinearRegression model
    2. Save the model to the specified path using joblib
    3. Print a success message
    4. Make predictions on the test set
    5. Calculate the cost using the custom cost function
    6. Print the cost and sample predictions
    """
    # Your code here
    pass


# --------- Main Program ---------
if __name__ == "__main__":
    # This section will be executed when the script is run directly
    # You can use it to test your functions
    pass
