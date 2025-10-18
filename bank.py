# Task 3: Decision Tree Classifier - Bank Marketing Dataset
# ----------------------------------------------------------

# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Load dataset
# (Make sure your file name matches — e.g., bank-full.csv or bank.csv)
data = pd.read_csv("bank-full.csv", sep=';')

print("\n--- Dataset Loaded ---")
print(data.head())
print("\nShape:", data.shape)

# Step 3: Check for missing values
print("\n--- Missing Values ---")
print(data.isnull().sum())

# Step 4: Encode categorical columns
le = LabelEncoder()
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = le.fit_transform(data[col])

print("\n--- After Encoding ---")
print(data.head())

# Step 5: Split dataset into features and target
X = data.drop('y', axis=1)
y = data['y']  # Target column (yes/no for term deposit)

# Step 6: Split into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train Decision Tree Classifier
clf = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# Step 8: Predictions
y_pred = clf.predict(X_test)

# Step 9: Evaluate model
print("\n--- Model Evaluation ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 10: Visualize the Decision Tree
plt.figure(figsize=(20,10))
plot_tree(clf, feature_names=X.columns, class_names=['No', 'Yes'], filled=True)
plt.title("Decision Tree Classifier - Bank Marketing Dataset")
plt.show()

# Step 11: Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(data.corr(), cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()
