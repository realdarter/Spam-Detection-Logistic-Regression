import pandas as pd

def read_CSV(csv_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file, sep=' ', quotechar='|', header=None)

    # Convert DataFrame to a list of lists
    data = df.values.tolist()

    return data
