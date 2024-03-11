# imports
import read
import numpy
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

row = str(read.read_CSV(TEST_FILE)[0])#.split(',')

print(len(row.split(",")))
print()



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

num_features = len(row.split(","))
WEIGHTS = initialize_weights(num_features)
#print("Initial Weights:", WEIGHTS)
#print(len(WEIGHTS))

# Implement the sigmoid function
def sigmoid(z):
    """
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
def helper_function():
    return -1



# The prediction function
# Takes a test instance as input and outputs the predicted label
# This function should call Helper function
def prediction_function(test_instance):
    return -1




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
    pass



# main Function
