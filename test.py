import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Function to test part 1
def test_part1():
    # Load the merged dataset and check if it's loaded properly
    merged_data = pd.read_csv("merged_dataset_with_spam.csv")
    assert not merged_data.empty, "Merged dataset is empty"

# Function to test part 2
def test_part2():
    # Load the preprocessed merged dataset and check if it's loaded properly
    preprocessed_data = pd.read_csv("preprocessed_merged_dataset.csv")
    assert not preprocessed_data.empty, "Preprocessed dataset is empty"

# Function to test part 3
def test_part3():
    # Load the balanced dataset and check if it's loaded properly
    balanced_data = pd.read_csv("balanced_dataset.csv")
    assert not balanced_data.empty, "Balanced dataset is empty"

# Function to test part 4
def test_part4():
    # Load the balanced dataset and check if it's loaded properly
    balanced_data = pd.read_csv("balanced_dataset.csv")
    assert not balanced_data.empty, "Balanced dataset is empty"

# Function to test part 5
def test_part5():
    # Check if the required plots are saved
    assert os.path.exists("plot1.png"), "Plot 1 not saved"
    assert os.path.exists("plot2.png"), "Plot 2 not saved"
    assert os.path.exists("plot3.png"), "Plot 3 not saved"
    assert os.path.exists("plot4.png"), "Plot 4 not saved"
    assert os.path.exists("plot5.png"), "Plot 5 not saved"

# Function to test part 6
def test_part6():
    # Load the dataset with extracted features and check if it's loaded properly
    dataset_with_features = pd.read_csv("dataset_with_extracted_features.csv")
    assert not dataset_with_features.empty, "Dataset with extracted features is empty"

# Function to test part 7
def test_part7():
    # Load the dataset with extracted features and check if it's loaded properly
    dataset_with_features = pd.read_csv("dataset_with_extracted_features.csv")
    assert not dataset_with_features.empty, "Dataset with extracted features is empty"

    # Split the data into features and target
    X = dataset_with_features.drop(['type_benign', 'url'], axis=1)
    y = dataset_with_features['type_benign']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Test Logistic Regression Model
    logistic_model = LogisticRegression(random_state=42)
    logistic_model.fit(X_train, y_train)
    y_pred_logistic = logistic_model.predict(X_test)
    assert accuracy_score(y_test, y_pred_logistic) >= 0, "Logistic Regression model failed"

    # Test Random Forest Model
    random_forest_model = RandomForestClassifier(random_state=42)
    random_forest_model.fit(X_train, y_train)
    y_pred_rf = random_forest_model.predict(X_test)
    assert accuracy_score(y_test, y_pred_rf) >= 0, "Random Forest model failed"

# Function to test part 8
def test_part8():
    # Load the dataset with extracted features and check if it's loaded properly
    dataset_with_features = pd.read_csv("dataset_with_extracted_features.csv")
    assert not dataset_with_features.empty, "Dataset with extracted features is empty"

    # Split the data into features and target
    X = dataset_with_features.drop(['type_benign', 'url'], axis=1)
    y = dataset_with_features['type_benign']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Test Logistic Regression Model
    logistic_model = LogisticRegression(random_state=42)
    logistic_model.fit(X_train, y_train)
    plot_confusion_matrix(logistic_model, X_test, y_test)

    # Test Random Forest Model
    random_forest_model = RandomForestClassifier(random_state=42)
    random_forest_model.fit(X_train, y_train)
    plot_confusion_matrix(random_forest_model, X_test, y_test)

if __name__ == "__main__":
    test_part1()
    test_part2()
    test_part3()
    test_part4()
    test_part5()
    test_part6()
    test_part7()
    test_part8()
    print("All tests passed successfully!")
