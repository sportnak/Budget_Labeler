# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.utils.class_weight import compute_sample_weight

import pandas as pd
import numpy as np
import string

from predict_data import predict_file


def clean_data(data):
    data['Description'] = data['Description'].astype(str)
    data['Description'] = data['Description'].str.lower()
    data['Description'] = data['Description'].fillna({ 'Description': '' })
    data['Description'] = data['Description'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
    data['Description'] = data['Description'].str.replace('\d+', '')

    data['Amount'] = data['Amount'].str.replace('\$', '')
    data['Amount'] = data['Amount'].str.replace(',', '')

    data['Is_Positive'] = data['Amount'].apply(lambda x: not '-' in x)

    data['Amount'] = data['Amount'].str.replace('-', '')
    return data


# Assuming you have a DataFrame with 'Text' column and 'Category' column (for supervised learning)
# X is the feature (Text) and y is the target variable (Category)

# Create a DataFrame with a single column named 'Text'
df = pd.read_csv('./crawford.csv')

# Display summary statistics to identify outliers
# print("Summary Statistics:")
# print(df.describe())
# print("Data Types:")
# print(df.dtypes)

# Find rows with null values
# Display rows with null values
# rows_with_nulls = df[df.isnull().any(axis=1)]
# print("\nRows with Null Values:")
# print(rows_with_nulls)
df = clean_data(df)


X = df[[ 'Description', 'Amount', 'Is_Positive', 'Account']]
Y = df['Category']




# Split the dataset into training and testing sets
# Summary statistics for the training dataset
print("Training Dataset Summary:")
print(X.describe())

# Convert Description data to numerical features using CountVectorizer
transformer = ColumnTransformer([
    ('vectorizer', CountVectorizer(), 'Description'),
    ('vectorizer2', CountVectorizer(), 'Account')
], remainder='passthrough')


X_train_vectorized = transformer.fit_transform(X)
sample_weights = compute_sample_weight('balanced',  y=Y)

# Initialize and train a simple model (Naive Bayes classifier in this case)
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train_vectorized, Y)

# Make predictions on the test set
# predictions = model.predict(X_test_vectorized)

# DEBUG A LABEL
# indices_to_debug = (y_test == 'Utilities')

# debug_text = X_test[indices_to_debug]['Description']
# debug_predictions = predictions[indices_to_debug]
# true_labels = y_test[indices_to_debug]

# for pred, true_label, text in zip(debug_predictions, true_labels, debug_text):
#     print(f"Predicted: {pred}, Actual: {true_label} Text: {text}")


# Evaluate the model
# accuracy = accuracy_score(Y_test, predictions)
# print(f"Model Accuracy: {accuracy}")

# # Display classification report for more detailed metrics
predict_file(lambda x: clean_data(x), model, transformer)

# Show feature importances
# feature_importances = model.feature_importances_
# print("Feature Importances:")
# for feature, importance in zip(X_train.columns, feature_importances):
#     print(f"{feature}: {importance}")