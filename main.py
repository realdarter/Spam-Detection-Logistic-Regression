# imports
import numpy as np
import math
import time

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
    Sigmoid activation function 😼
    
    Parameters:
    z (float): The raw input to the sigmoid function
    
    Returns:
    float: Output of the sigmoid function
                  |\_ |\_    [  Sigma Cat :3 ]
                   \` .. \   )/   
              __,.-" =__Y=         
            ."        )
      _    /   ,    \/\_
     ((____|    )_-\ \_-`
     `-----'`-----` `--`
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
    # calculate the raw input to the sigmoid function
    z = 0
    for w, x in zip(weights, test_instance):
        z += w * x
        #print(f"[|{str(z)} + w:{w} + x:{x}|] ", end="")

    # call the sigmoid function
    probability = sigmoid(z)
    #print(probability)
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
    # use the helper function to get the probability
    probability = helper_function(weights, test_instance)
    # compare with the decision threshold return 1 or 0
    if (probability >= threshold):
        return 1
    return 0




# This function takes a test set as input, call the predict function to predict a label for it,
# and prints the accuracy, P, R, and F1 score of the positive class and negative class and the confusion matrix
def evaluate_model(test_set, weights, threshold=0.5):
    """
    🤬😵‍💫🫨🗿 Evaluate the Logistic Regression model on a test set 🤬😵‍💫🫨🗿

    Parameters:
    test_set (tuple): A tuple containing feature names and the test data matrix.
    weights (list): The learned weight vector.
    threshold (float): Decision threshold for classification.

    Returns:
    tuple: A tuple containing accuracy, precision, recall, F1 score, and confusion matrix.
    """
    feature_names, test_data = test_set
    num_features = len(feature_names) - 1

    true_positive, false_positive, true_negative, false_negative = 0, 0, 0, 0

    for instance in test_data:
        features = instance[:-1]
        label = instance[-1]

        # use the learned weights to make a prediction here
        prediction = prediction_function(weights, features, threshold)

        # update confusion matrix
        if label == 1 and prediction == 1:
            true_positive += 1
        elif label == 0 and prediction == 1:
            false_positive += 1
        elif label == 0 and prediction == 0:
            true_negative += 1
        elif label == 1 and prediction == 0:
            false_negative += 1

    # calculate  evaluation metrics
    accuracy = (true_positive + true_negative) / len(test_data)
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    confusion_matrix = {
        'True Positive': true_positive,
        'False Positive': false_positive,
        'True Negative': true_negative,
        'False Negative': false_negative
    }

    return accuracy, precision, recall, f1_score, confusion_matrix

def compute_log_likelihood(weights, data_matrix):
    """
    Compute the log-likelihood for the Logistic Regression model. 😬

    Parameters:
    weights (list): The weight vector.
    data_matrix (numpy.ndarray): The data matrix.

    Returns:
    float: The log-likelihood.
    """
    log_likelihood = 0.0

    for instance in data_matrix:
        features = instance[:-1]
        label = instance[-1]

        # calculate the raw input to the sigmoid function here
        z = np.dot(weights, features)

        # calculate the log-likelihood contribution for the current  instance
        log_likelihood += label * z - np.log(1 + np.exp(z))

    return log_likelihood


# Train the Logistic Regression in a function using Stochastic Gradient Descent
# Also compute the log-oss in this function
def train_logistic_regression(train_set, learning_rate, iterations):
    """
    Train the Logistic Regression model using Stochastic Gradient Descent. 😵

    Parameters:
    train_set (tuple): A tuple containing feature names and the data matrix.
    learning_rate (float): The learning rate for gradient descent.
    iterations (int): The number of iterations for training.

    Returns:
    list: The learned weight vector.
    """
    feature_names, data_matrix = train_set
    num_features = len(feature_names) - 1
    weights = initialize_weights(num_features)
    
    # Inside the train_logistic_regression function
    for iteration in range(iterations):
        # Shuffle the data to introduce stochasticity
        #np.random.shuffle(data_matrix)
        for instance in data_matrix:
            features = instance[:-1]
            label = instance[-1]

            prediction = helper_function(weights, features)
            
            # Vectorized weight update
            gradient = (prediction - label) * features
            weights -= learning_rate * gradient
        # Log-likelihood calculation after each epoch
        log_likelihood = compute_log_likelihood(weights, data_matrix)
        print(f"Iteration {iteration + 1}/{iterations}, Log-Likelihood: {log_likelihood}")
    return weights


# Function to read the input dataset
def read_input_dataset(file_path):
    """
    Read the input dataset from the given file path. 🥰

    Parameters:
    file_path (str): The path to the input dataset file.

    Returns:
    tuple: A tuple containing feature names and data matrix.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # extract feature names from the first line
    feature_names = lines[0].strip().split(',')
    
    data_lines = []
    # extract data from the remaining lines
    for line in lines[1:]:
        data_lines.append(line.strip().split(','))

    # convert data to a NumPy array
    data_matrix = np.array(data_lines, dtype=float)
    return feature_names, data_matrix

def print_running_time(start_time):
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print("--- Runtime: %d hours, %d minutes, %d seconds ---" % (hours, minutes, seconds))

def main():
    # load training and test sets
    train_set = read_input_dataset(TRAIN_FILE)
    test_set = read_input_dataset(TEST_FILE)
    print("Succesfully read Data!")
    start_time = time.time()

    # train the Logistic Regression model
    learned_weights = train_logistic_regression(train_set, RATE, ITERATIONS)

    # evaluate the model on the test set
    accuracy, precision, recall, f1_score, confusion_matrix = evaluate_model(test_set, learned_weights)

    print_running_time(start_time)

    # printing the evaluation metrics
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1_score}")
    print("Confusion Matrix:")
    for key, value in confusion_matrix.items():
        print(f"{key}: {value}")
    input("Enter to Finish: ")

if __name__ == "__main__":
    main()
