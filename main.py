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
    tr_accuracy, tr_pos_precision, tr_pos_recall, tr_pos_f1_Score, tr_neg_precision, tr_neg_recall, tr_neg_f1_Score, tr_confusion_matrix = evaluate_model(train_set, learned_weights) # evaluate the model on the train set
    te_accuracy, te_pos_precision, te_pos_recall, te_pos_f1_Score, te_neg_precision, te_neg_recall, te_neg_f1_Score, te_confusion_matrix = evaluate_model(test_set, learned_weights) # evaluate the model on the test set

    print_running_time(start_time)

    total_log_loss = 1/ITERATIONS*np.sum(log_loss_array)

    # printing the evaluation metrics
    print("-------Evaluation Metrics-------")
    print("[Train_Set]")
    print(f"Accuracy: {tr_accuracy}")
    print(f"Total cost of log loss over {ITERATIONS} iterations:", total_log_loss)

    print(f"Positive Class (Spam)")
    print(f"Precision: {tr_pos_precision}")
    print(f"Recall: {tr_pos_recall}")
    print(f"F1 Score: {tr_pos_f1_Score}\n")
 
    print(f"Negative Class (Ham)")
    print(f"Precision: {tr_neg_precision}")
    print(f"Recall: {tr_neg_recall}")
    print(f"F1 Score: {tr_neg_f1_Score}\n")

    print("Confusion Matrix:")
    for key, value in tr_confusion_matrix.items():
        if (re.findall(r'\bpositive\b', key, flags=re.IGNORECASE)):
            print(f"{key} (Spam) : {value}")
        else:
            print(f"{key} (Ham) : {value}")
    print("--------------------------------\n")

    print("-------Evaluation Metrics-------")
    print("[Test_Set]")
    print(f"Accuracy: {te_accuracy}")
    print(f"Total Log Loss: {log_loss_array[ITERATIONS-1]}\n")

    print(f"Positive Class (Spam)")
    print(f"Precision: {te_pos_precision}")
    print(f"Recall: {te_pos_recall}")
    print(f"F1 Score: {te_pos_f1_Score}\n")
 
    print(f"Negative Class (Ham)")
    print(f"Precision: {te_neg_precision}")
    print(f"Recall: {te_neg_recall}")
    print(f"F1 Score: {te_neg_f1_Score}\n")

    print("Confusion Matrix:")
    for key, value in te_confusion_matrix.items():
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
