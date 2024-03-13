from logistic_regression import *
from machine_learning_analysis import *
import numpy as np

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
    accuracy, precision, recall, f1_score, confusion_matrix = evaluate_model(test_set, learned_weights) # evaluate the model on the test set

    print_running_time(start_time)

    # printing the evaluation metrics
    print("------------Evaluation Metrics------------")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1_score}")
    print("Confusion Matrix:")
    for key, value in confusion_matrix.items():
        print(f"{key}: {value}")
    print("------------------------------------------")

    print("Finished.")
    
    #Display
    try:
        plot_line_chart(np.arange(ITERATIONS), log_loss_array, label='Sample Data', xlabel='Iterations', ylabel='Log Loss', title='Log Loss vs Iterations')
    except ValueError:
        print(f"Error Creating Line Chart: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
