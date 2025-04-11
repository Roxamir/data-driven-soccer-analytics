import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
from sklearn.model_selection import cross_val_score

# Read the CSV file
filename = "Datasets/playerdata/api_cleaned.csv"
df = pd.read_csv(filename)

print("\nOriginal Dataset:")
print(df)

# Remove rows with NaN values
df = df.dropna()

# Calculate per 90 minutes metrics
df['key_passes_per90'] = df['key_passes'] / (df['minutes_played'] / 90)
df['goals_per90'] = df['goals'] / (df['minutes_played'] / 90)
df['assists_per90'] = df['assists'] / (df['minutes_played'] / 90)
df['shots_on_target_per90'] = df['shots_on_target'] / (df['minutes_played'] / 90)

# Select only the per 90 metrics
important_columns = ['key_passes_per90', 'goals_per90', 'assists_per90', 'shots_on_target_per90']

# Select features and target
X = df[important_columns]
y = df['position']  # Assuming this is the target variable

print("\n\nCleaned dataset:\n", X)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n\nTraining dataset:\n", X_train)
print("\n\nTraining labels:\n", y_train)
print("\n\nTesting dataset:\n", X_test)
print("\n\nTesting labels:\n", y_test)

# Create and train the decision tree classifier with reduced complexity
clf = DecisionTreeClassifier(
    random_state=42,
    max_depth=5,  # Limit the maximum depth of the tree
    min_samples_split=10,  # Minimum number of samples required to split an internal node
    min_samples_leaf=5,  # Minimum number of samples required to be at a leaf node
    max_leaf_nodes=10  # Limit the total number of leaf nodes
)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualize the confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
classes = sorted(y.unique())  # Get unique position labels

# Create heatmap with actual labels
sns.heatmap(cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues', 
            xticklabels=classes, 
            yticklabels=classes)
plt.title('Confusion Matrix of Player Positions')
plt.xlabel('Predicted Position')
plt.ylabel('True Position')
plt.tight_layout()
plt.show()

# Visualize the decision tree
plt.figure(figsize=(20,10))
tree.plot_tree(clf, feature_names=important_columns, class_names=clf.classes_, filled=True, rounded=True)
plt.title('Decision Tree Visualization')
plt.tight_layout()
plt.show()

# Calculate and print overall accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nOverall Accuracy: {accuracy:.2%}")

# Perform cross-validation for more robust accuracy estimation
cv_scores = cross_val_score(clf, X, y, cv=5)
print("\nCross-Validation Scores:", cv_scores)
print(f"Mean Cross-Validation Accuracy: {cv_scores.mean():.2%}")
print(f"Standard Deviation of Cross-Validation Scores: {cv_scores.std():.2%}")

# Additional performance metrics
print("\nDetailed Classification Metrics:")
print(classification_report(y_test, y_pred, target_names=classes))

# Print feature importances
feature_importance = pd.DataFrame({
    'feature': important_columns,
    'importance': clf.feature_importances_
}).sort_values('importance', ascending=False)
print("\nFeature Importances:")
print(feature_importance)

# Visualizations for Performance Metrics by Position

# 1. Bar Graph: Average Performance Metrics by Position
plt.figure(figsize=(12, 6))
df_grouped = df.groupby('position')[important_columns].mean()
df_grouped.plot(kind='bar', ax=plt.gca())
plt.title('Average Performance Metrics by Position (per 90 minutes)')
plt.xlabel('Position')
plt.ylabel('Metric Value')
plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 2. Box Plot: Distribution of Metrics Across Positions
plt.figure(figsize=(12, 6))
df_melted = df.melt(id_vars=['position'], value_vars=important_columns, 
                    var_name='Metric', value_name='Value')
sns.boxplot(x='position', y='Value', hue='Metric', data=df_melted)
plt.title('Distribution of Performance Metrics by Position')
plt.xlabel('Position')
plt.ylabel('Metric Value (per 90 minutes)')
plt.xticks(rotation=45)
plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 3. Violin Plot: Distribution of Goals and Key Passes by Position
plt.figure(figsize=(14, 8))

# Create a subplot grid
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Violin plot for Key Passes per 90 minutes
sns.violinplot(x='position', y='key_passes_per90', data=df, ax=ax1)
ax1.set_title('Distribution of Key Passes by Position')
ax1.set_xlabel('Position')
ax1.set_ylabel('Key Passes per 90 minutes')
ax1.tick_params(axis='x', rotation=45)

# Violin plot for Goals per 90 minutes
sns.violinplot(x='position', y='goals_per90', data=df, ax=ax2)
ax2.set_title('Distribution of Goals by Position')
ax2.set_xlabel('Position')
ax2.set_ylabel('Goals per 90 minutes')
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
