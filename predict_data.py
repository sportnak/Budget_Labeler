import csv
import numpy as np
import pandas as pd

def write_to_csv(file_path, data, header=None):
    with open(file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        # Write the header if provided
        if header:
            csv_writer.writerow(header)

        # Write the data
        csv_writer.writerows(data)


def predict_file(clean_data, model, transformer):
    test_df = pd.read_csv('./crawford_test.csv')
    X_test = clean_data(test_df.copy())

    X_test_vectorized = transformer.transform(X_test)
    class_probabilities = model.predict_proba(X_test_vectorized)
    class_order = model.classes_
    # print(np.where(class_order == "Groceries"))

    result = []
    header = ['Description', 'Category', 'Confidence']
    i = 0
    for row in class_probabilities:
        max_index = np.argmax(row)
        result.append([test_df.at[i, 'Description'], class_order[max_index], row[max_index]])
        i+=1
        
    write_to_csv("./output.csv", result, header)