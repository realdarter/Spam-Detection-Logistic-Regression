"""
 * Description: This code runs logistic regression with stochastic gradient descent for binary classification. 
 * It includes functions for model training, evaluation, and log loss calculation, 
 * allows for a customizable and interpretable machine learning approach.
 * Author names: Goose
 * Last modified date: 3/13/2024
 * Creation date: 3/10/2024
"""

from logistic_regression import *
from analysis import *
import numpy as np
import re

# Directories
TEST_FILE = 'data/test.csv'
TRAIN_FILE = 'data/train.csv'
# Learning Rate
RATE = 0.01
# The Weights to Learn
WEIGHTS = []
# Number of Iterations
ITERATIONS = 200

def print_running_time(start_time):
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print("--- Runtime: %d hours, %d minutes, %d seconds ---" % (hours, minutes, seconds))

def main():
    # load training and test sets
    try:
        train_set = read_input_dataset(TRAIN_FILE)
        test_set = read_input_dataset(TEST_FILE)
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    print("Succesfully read Data!")
    start_time = time.time()


    learned_weights, log_loss_array = train_logistic_regression(train_set, RATE, ITERATIONS) # train the Logistic Regression model
    accuracy, pos_precision, pos_recall, pos_f1_Score, neg_precision, neg_recall, neg_f1_Score, confusion_matrix = evaluate_model(test_set, learned_weights) # evaluate the model on the test set

    print_running_time(start_time)

    # printing the evaluation metrics
    print("-------Evaluation Metrics-------")
    print(f"Accuracy: {accuracy}")
    print(f"Total Log Loss: {log_loss_array[ITERATIONS-1]}\n")

    print(f"Positive Class (Spam)")
    print(f"Precision: {pos_precision}")
    print(f"Recall: {pos_recall}")
    print(f"F1 Score: {pos_f1_Score}\n")
 
    print(f"Negative Class (Ham)")
    print(f"Precision: {neg_precision}")
    print(f"Recall: {neg_recall}")
    print(f"F1 Score: {neg_f1_Score}\n")

    print("Confusion Matrix:")
    for key, value in confusion_matrix.items():
        if (re.findall(r'\bpositive\b', key, flags=re.IGNORECASE)):
            print(f"{key} (Spam) : {value}")
        else:
            print(f"{key} (Ham) : {value}")
    print("--------------------------------")

    print("Finished")
    
    #Display
    try:
        plot_line_chart(np.arange(ITERATIONS), log_loss_array, label='Sample Data', xlabel='Iterations', ylabel='Log Loss', title='Log Loss vs Iterations')
    except ValueError:
        print(f"Error Creating Line Chart: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
