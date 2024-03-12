# imports
import read
import numpy as np
import math

# variables
TEST_FILE = 'data/test.csv'
TRAIN_FILE = 'data/train.csv'
# Learning Rate
RATE = 0.01
# The Weights to Learn
WEIGHTS = []
# Number of Iterations
ITERATIONS = 200

# Constructor initializes the weight vector. Initialize it by setting it to the 0 vector.
def initialize_weights(num_features):
    """
    Initialize the weight vector by setting it to the 0 vector.
    
    Parameters:
    num_features (int): Number of features in the weight vector
    
    Returns:
    list: The initialized weight vector
    """
    return [0.0] * num_features


#print("Initial Weights:", WEIGHTS)
#print(len(WEIGHTS))



# Implement the sigmoid function
def sigmoid(z):
    """
    🗿🗿🗿🗿🗿🗿🗿🗿
    Sigmoid activation function
    
    Parameters:
    z (float): The raw input to the sigmoid function
    
    Returns:
    float: Output of the sigmoid function
    """
    return 1 / (1 + math.exp(-z))



# Helper function for prediction
# Takes a test instance as input and outputs the probability of the label being 1
# This function should call sigmoid()
def helper_function(weights, test_instance):
    """
    Helper function for prediction.
    
    Parameters:
    weights (list): The weight vector
    test_instance (list): The feature values of the test instance
    
    Returns:
    float: Probability of the label being 1
    """
    # Calculate the raw input to the sigmoid function
    z = 0
    for w, x in zip(weights, test_instance):
        z += w * x
        #print(f"[|{str(z)} + w:{w} + x:{x}|] ", end="")

    # Call the sigmoid function
    probability = sigmoid(z)
    print(probability)
    return probability


# The prediction function
# Takes a test instance as input and outputs the predicted label
# This function should call Helper function
def prediction_function(weights, test_instance, threshold=0.5):
    """
    Prediction function.
    
    Parameters:
    weights (list): The weight vector
    test_instance (list): The feature values of the test instance
    threshold (float): Decision threshold for classification
    
    Returns:
    int: Predicted label (0 or 1)
    """
    # Use the helper function to get the probability
    probability = helper_function(weights, test_instance)
    # Compare with the decision threshold
    if (probability >= threshold):
        return 1
    return 0




# This function takes a test set as input, call the predict function to predict a label for it,
# and prints the accuracy, P, R, and F1 score of the positive class and negative class and the confusion matrix
def evaluate_model(test_set):
    pass



# Train the Logistic Regression in a function using Stochastic Gradient Descent
# Also compute the log-oss in this function
def train_logistic_regression(train_set, learning_rate, iterations):
    pass


# Function to read the input dataset
def read_input_dataset(file_path):
    """
    Read the input dataset from the given file path.

    Parameters:
    file_path (str): The path to the input dataset file.

    Returns:
    tuple: A tuple containing feature names and data matrix.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Extract feature names from the first line
    feature_names = lines[0].strip().split(',')
    
    data_lines = []
    # Extract data from the remaining lines
    for line in lines[1:]:
        data_lines.append(line.strip().split(','))

    # Convert data to a NumPy array
    data_matrix = np.array(data_lines, dtype=float)

    return feature_names, data_matrix


feature_names, data_matrix = read_input_dataset(TRAIN_FILE)
print(feature_names)
print(type(feature_names))

num_features = len(feature_names)
print(num_features)
WEIGHTS = initialize_weights(num_features)
probability = helper_function(WEIGHTS, data_matrix[2])
print("Probability:", probability)
prediction = prediction_function(WEIGHTS, data_matrix[2], threshold=0.5)
print("Prediction:", prediction)

# main Function




#sigma()