# imports
import numpy as np
import math
import time

# Constructor initializes the weight vector. Initialize it by setting it to the 0 vector.
def initialize_weights(num_features):
    """
    Initialize the weight vector by setting it to the 0 vector.
    
    Parameters: num_features (int): Number of features in the weight vector
    Returns: list: The initialized weight vector
    """
    return [0.0] * num_features


#print("Initial Weights:", WEIGHTS)
#print(len(WEIGHTS))

# Implement the sigmoid function
def sigmoid(z):
    """
    Sigmoid activation function 😼
    
    Parameters: z (float): The raw input to the sigmoid function
    Returns: float: Output of the sigmoid function

                   |\_ |\    [  Sigma Cat :3 ]
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
def helper_function(weights, test_matrix):
    """
    Helper function for prediction.
    
    Parameters: weights (numpy.ndarray): The weight vector. test_matrix (numpy.ndarray): The feature values of the test instances.
    Returns: numpy.ndarray: Probabilities of the labels being 1 for each instance.
    """
    z = np.dot(test_matrix, weights)
    return sigmoid(z)


# The prediction function
# Takes a test instance as input and outputs the predicted label
# This function should call Helper function
def prediction_function(weights, test_instance, threshold=0.5):
    """
    Prediction function.
    
    Parameters: weights (list): The weight vector. test_instance (list): The feature values of the test instance
    threshold (float): Decision threshold for classification
    Returns: int: Predicted label (0 or 1)
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
    evaluates the Logistic Regression model on a test set 

    Parameters: test_set (tuple): A tuple containing feature names and the test data matrix. weights (list): The learned weight vector. 
    threshold (float): Decision threshold for classification.
    tuple: A tuple containing precision, recall, F1-score, and confusion matrix for both positive and negative classes.
    """
    _, test_data = test_set #skipping features and getting test data only
    
    true_positive, false_positive, true_negative, false_negative = 0, 0, 0, 0

    for instance in test_data:
        features = instance[:-1]
        label = instance[-1]

        # use the learned weights to make a prediction here
        prediction = prediction_function(weights, features, threshold)

        if label == 1 and prediction == 1:
            true_positive += 1
        elif label == 0 and prediction == 1:
            false_positive += 1
        elif label == 0 and prediction == 0:
            true_negative += 1
        elif label == 1 and prediction == 0:
            false_negative += 1
    
    accuracy = (true_positive + true_negative) / len(test_data)
    # Positive class
    pos_precision = true_positive/(true_positive + false_positive)
    pos_recall = true_positive / (true_positive + false_negative)
    pos_f1_Score = 2 * (pos_precision * pos_recall) / (pos_precision + pos_recall)
    # Negative class
    neg_precision = true_negative / (true_negative + false_positive)
    neg_recall = true_negative / (true_negative + false_positive)
    neg_f1_score = 2 * (neg_precision * neg_recall) / (neg_precision + neg_recall)


    confusion_matrix = {
        'True Positive': true_positive,
        'False Positive': false_positive,
        'True Negative': true_negative,
        'False Negative': false_negative
    }

    return accuracy, pos_precision, pos_recall, pos_f1_Score, neg_precision, neg_recall, neg_f1_score, confusion_matrix

def compute_log_loss(true_labels, predicted_probabilities):
    """
    Computes the log loss for binary classification.

    Parameters: y_true (numpy.ndarray): True labels. y_pred (numpy.ndarray): Predicted probabilities.
    Returns: float: The log loss.
    """
    num_instances = len(true_labels)
    
    # the clip predictions to avoid log(0) or log(1)
    clipped_predictions = np.clip(predicted_probabilities, 1e-15, 1 - 1e-15) # np.clip bounds the current predicted probabilities to avoid extreme values (0 or 1), 
    # this then prevents a numerical instability in the logistic regression computations
    
    # compute the log loss here
    loss = -1/num_instances * np.sum(true_labels * np.log(clipped_predictions) + (1 - true_labels) * np.log(1 - clipped_predictions)) #-1/N Sumation{N,i=1}(Yi * log(pi) + (1-yi) * log(1-pi))
    return loss


# Train the Logistic Regression in a function using Stochastic Gradient Descent
# Also compute the log-oss in this function
def train_logistic_regression(train_set, learning_rate, iterations):
    """
    trains the Logistic Regression model using Stochastic Gradient Descent.

    Parameters: train_set (tuple): A tuple containing feature names and the data matrix. learning_rate (float): The learning rate for gradient descent.
    iterations (int): The number of iterations for training.
    Returns: list: The learned weight vector and list: Log Loss Array.
    """
    start_time = time.time()
    feature_names, data_matrix = train_set
    num_features = len(feature_names) - 1
    weights = initialize_weights(num_features)
    log_loss_array = []

    # Track true labels and predicted probabilities during training
    y_true_list, y_pred_list = [], []
    
    # inside the train_logistic_regression function
    for iteration in range(iterations):
        for instance in data_matrix:
            features = instance[:-1]
            label = instance[-1]

            prediction = helper_function(weights, features)
            
            # Vectorized weight update
            gradient = (prediction - label) * features
            weights -= learning_rate * gradient
        
        # Calculate predicted probabilities on the entire dataset after each epoch
        y_true = data_matrix[:, -1]
        y_pred = [helper_function(weights, features) for features in data_matrix[:, :-1]]
        
        # Append to lists for later log loss calculation
        y_true_list.extend(y_true)
        y_pred_list.extend(y_pred)
        
        # Log-likelihood calculation after each epoch
        log_loss = compute_log_loss(np.array(y_true_list), np.array(y_pred_list))
        log_loss_array.append(log_loss)
        print(f"Iteration {iteration + 1}/{iterations}".ljust(30), f"Log Loss: {log_loss}".ljust(40),f"Elapsed Time: {time.time() - start_time}")
    
    return weights, log_loss_array


# Function to read the input dataset
def read_input_dataset(file_path):
    """
    Read the input dataset from the given file path.

    Parameters: file_path (str): The path to the input dataset file.
    Returns: tuple: A tuple containing feature names and data matrix.
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