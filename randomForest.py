import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load features for NSCLC (and possibly normal data if you have it)
df_nslc = pd.read_excel("combined_file.xlsx")
df_nslc.dropna(inplace=True)  # Drop any rows with missing values
print(df_nslc.shape)
# Assuming you also have normal protein data in a separate Excel file
# df_normal = pd.read_excel("Normal_Human_Protein_Sequences.xlsx")

# For simplicity, we'll use just the NSCLC data here
X = df_nslc.drop(columns=["Disease_Index"])  # Features
y = df_nslc["Disease_Index"]  # Target labels (1 for NSCLC)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Generate classification report and confusion matrix
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)

# Visualize confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for NSCLC Classifier')
plt.show()
