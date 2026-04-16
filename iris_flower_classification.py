# ============================================================
#   TASK 1: IRIS FLOWER CLASSIFICATION
#   AICTE OASIS Data Science Internship
# ============================================================
#
#   What this project does:
#   - Loads the Iris dataset (3 flower species, 150 samples)
#   - Explores and visualizes the data
#   - Trains a Machine Learning model (Random Forest)
#   - Evaluates its accuracy
#   - Predicts the species of a new flower
#
# ============================================================

# ── STEP 1: Install required libraries (run once in terminal)
# pip install pandas numpy matplotlib seaborn scikit-learn

# ── STEP 2: Import all necessary libraries
import pandas as pd                          # For data tables
import numpy as np                           # For numbers & arrays
import matplotlib.pyplot as plt              # For graphs
import seaborn as sns                        # For beautiful graphs

from sklearn.datasets import load_iris       # Built-in Iris dataset
from sklearn.model_selection import train_test_split  # Split data
from sklearn.ensemble import RandomForestClassifier   # ML Model
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("✅ All libraries imported successfully!\n")


# ────────────────────────────────────────────────
# STEP 3: Load the Dataset
# ────────────────────────────────────────────────

iris = load_iris()   # Load the dataset from scikit-learn

# Convert to a Pandas DataFrame (like a spreadsheet table)
df = pd.DataFrame(
    data=iris.data,
    columns=iris.feature_names   # sepal length, sepal width, petal length, petal width
)
df['species'] = iris.target                          # 0, 1, 2 (numbers)
df['species_name'] = df['species'].map(              # Map numbers to names
    {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
)

print("📊 Dataset Loaded!")
print(f"   Total rows    : {len(df)}")
print(f"   Total columns : {len(df.columns)}")
print(f"   Species       : {df['species_name'].unique()}\n")


# ────────────────────────────────────────────────
# STEP 4: Explore the Data
# ────────────────────────────────────────────────

print("─" * 50)
print("FIRST 5 ROWS OF THE DATASET:")
print("─" * 50)
print(df.head())

print("\n─" * 50)
print("BASIC STATISTICS (mean, min, max, etc.):")
print("─" * 50)
print(df.describe())

print("\n─" * 50)
print("HOW MANY FLOWERS PER SPECIES:")
print("─" * 50)
print(df['species_name'].value_counts())
print()


# ────────────────────────────────────────────────
# STEP 5: Visualize the Data
# ────────────────────────────────────────────────

# Graph 1: Pairplot — See how features relate to each other
print("📈 Creating visualizations... (close each graph window to continue)")

sns.pairplot(df, hue='species_name', palette='Set2')
plt.suptitle("Pairplot of Iris Features by Species", y=1.02, fontsize=14)
plt.tight_layout()
plt.savefig("iris_pairplot.png", dpi=100, bbox_inches='tight')
plt.show()

# Graph 2: Box plot — See the spread of each feature
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
features = iris.feature_names

for i, ax in enumerate(axes.flatten()):
    df.boxplot(column=features[i], by='species_name', ax=ax)
    ax.set_title(features[i])
    ax.set_xlabel("Species")
    ax.set_ylabel("cm")

plt.suptitle("Box Plot of Features by Species", fontsize=14)
plt.tight_layout()
plt.savefig("iris_boxplot.png", dpi=100, bbox_inches='tight')
plt.show()

print("✅ Graphs saved as 'iris_pairplot.png' and 'iris_boxplot.png'\n")


# ────────────────────────────────────────────────
# STEP 6: Prepare Data for Training
# ────────────────────────────────────────────────

X = df[iris.feature_names]   # Features (inputs)  → measurements
y = df['species']             # Target  (output)   → species number

# Split: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% goes to testing
    random_state=42      # So results are the same every run
)

print(f"🔀 Data Split:")
print(f"   Training samples : {len(X_train)}")
print(f"   Testing  samples : {len(X_test)}\n")


# ────────────────────────────────────────────────
# STEP 7: Train the Machine Learning Model
# ────────────────────────────────────────────────

# Random Forest = many decision trees voting together
model = RandomForestClassifier(
    n_estimators=100,    # 100 decision trees
    random_state=42
)

model.fit(X_train, y_train)   # 🧠 Train the model!

print("🌲 Random Forest Model trained successfully!\n")


# ────────────────────────────────────────────────
# STEP 8: Evaluate the Model
# ────────────────────────────────────────────────

y_pred = model.predict(X_test)   # Make predictions on test data

accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Model Accuracy: {accuracy * 100:.2f}%\n")

print("─" * 50)
print("DETAILED CLASSIFICATION REPORT:")
print("─" * 50)
print(classification_report(
    y_test, y_pred,
    target_names=iris.target_names
))

# Graph 3: Confusion Matrix — How many predictions were correct?
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)
plt.title("Confusion Matrix", fontsize=14)
plt.xlabel("Predicted Species")
plt.ylabel("Actual Species")
plt.tight_layout()
plt.savefig("iris_confusion_matrix.png", dpi=100)
plt.show()

print("✅ Confusion matrix saved as 'iris_confusion_matrix.png'\n")


# ────────────────────────────────────────────────
# STEP 9: Feature Importance
# ────────────────────────────────────────────────

importances = model.feature_importances_
feature_df = pd.DataFrame({
    'Feature': iris.feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print("─" * 50)
print("WHICH FEATURE IS MOST IMPORTANT?")
print("─" * 50)
print(feature_df.to_string(index=False))
print()

plt.figure(figsize=(8, 4))
sns.barplot(
    data=feature_df,
    x='Importance',
    y='Feature',
    palette='viridis'
)
plt.title("Feature Importance in Random Forest", fontsize=13)
plt.tight_layout()
plt.savefig("iris_feature_importance.png", dpi=100)
plt.show()


# ────────────────────────────────────────────────
# STEP 10: Predict a NEW Flower! 🌸
# ────────────────────────────────────────────────

print("─" * 50)
print("PREDICTING A NEW FLOWER:")
print("─" * 50)

# Enter measurements for a new flower (in cm)
new_flower = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=iris.feature_names)
#                          sepal_len sepal_wid petal_len petal_wid

predicted_number = model.predict(new_flower)[0]
predicted_name   = iris.target_names[predicted_number]
probabilities    = model.predict_proba(new_flower)[0]

print(f"   Input measurements : {new_flower.values[0]}")
print(f"   Predicted species  : ✅ {predicted_name.upper()}")
print(f"   Confidence         : {max(probabilities) * 100:.1f}%")
print(f"   All probabilities  : setosa={probabilities[0]:.2f}, "
      f"versicolor={probabilities[1]:.2f}, virginica={probabilities[2]:.2f}")

print("\n" + "=" * 50)
print("🎉 PROJECT COMPLETE!")
print("   Files saved:")
print("   📊 iris_pairplot.png")
print("   📊 iris_boxplot.png")
print("   📊 iris_confusion_matrix.png")
print("   📊 iris_feature_importance.png")
print("=" * 50)
