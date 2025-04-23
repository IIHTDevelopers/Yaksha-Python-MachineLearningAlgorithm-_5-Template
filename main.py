import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE


# ---------------------------------- #
# 1. Load and Preprocess the Dataset
# ---------------------------------- #
def load_and_prepare_data(path="credit_risk_dataset.csv"):
    """
    Load and prepare the credit risk dataset.
    
    TODO:
    1. Load the CSV file from the given path
    2. Fill missing values using forward fill method
    3. Encode categorical features using LabelEncoder
    4. Print a success message
    5. Return the processed DataFrame
    """
    # Your code here
    pass


# ---------------------------- #
# 2. Apply SMOTE to balance classes
# ---------------------------- #
def apply_smote(X, y):
    """
    Apply SMOTE to balance the classes in the dataset.
    
    TODO:
    1. Create a SMOTE object with random_state=42
    2. Apply fit_resample to get balanced X and y
    3. Print a success message
    4. Return the resampled X and y
    """
    # Your code here
    pass


# --------------------------------- #
# 3. Hypothesis Function (SVM scores)
# --------------------------------- #
def hypothesis(model, X):
    """
    Calculate the decision function scores for the SVM model.
    
    TODO:
    1. Use the model's decision_function method to get raw scores
    2. Return the decision scores
    """
    # Your code here
    pass


# ------------------------------- #
# 4. Custom Cost Function (Hinge Loss)
# ------------------------------- #
def hinge_loss(y_true, scores):
    """
    Calculate the hinge loss for SVM predictions.
    
    TODO:
    1. Convert binary labels to signed format (-1, 1)
    2. Calculate the hinge loss using the formula: max(0, 1 - y * scores)
    3. Return the mean loss
    """
    # Your code here
    pass


# -------------------------------- #
# 5. Train the Model and Predict
# -------------------------------- #
def train_and_predict(df):
    """
    Train an SVM model and make predictions.
    
    TODO:
    1. Split the data into features (X) and target (y)
    2. Scale the features using StandardScaler
    3. Apply SMOTE to balance the classes
    4. Split the data into training and testing sets
    5. Create and train an SVM model
    6. Calculate decision scores using the hypothesis function
    7. Make predictions
    8. Calculate the hinge loss
    9. Print sample predictions and the loss
    """
    # Your code here
    pass


# ---------------- #
# Main Execution
# ---------------- #
if __name__ == "__main__":
    # This section will be executed when the script is run directly
    # You can use it to test your functions
    pass
