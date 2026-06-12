import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 80)
print("IRIS FLOWER CLASSIFICATION - COMPLETE PROJECT")
print("=" * 80)

# ============================================================================
# STEP 1: USE MEASUREMENTS OF IRIS FLOWERS AS INPUT DATA
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: LOAD AND EXPLORE DATA (Measurements)")
print("=" * 80)

# Load CSV file
df = pd.read_csv(r'C:\Users\ravi\Downloads\iris.csv')

print("\n✓ Dataset Loaded Successfully!")
print(f"Total Samples: {len(df)}")

# Extract measurements (features) - X
X = df.iloc[:, :-1]  # All columns except last (sepal length, sepal width, petal length, petal width)
print(f"\nFeatures (Measurements) Shape: {X.shape}")
print(f"Feature Names: {X.columns.tolist()}")

# Extract species labels (target) - y
y = df.iloc[:, -1]  # Last column
print(f"\nTarget (Species) Shape: {y.shape}")
print(f"Species Names: {y.unique()}")

# Show sample data
print("\nSample Data (First 5 rows):")
print(df.head())

# Show statistics
print("\nMeasurements Statistics:")
print(X.describe())

# ============================================================================
# STEP 2 & 3: TRAIN MACHINE LEARNING MODEL USING SCIKIT-LEARN
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2 & 3: PREPARE DATA AND TRAIN MODEL (80% Train, 20% Test)")
print("=" * 80)

# Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\n✓ Data Split Done!")
print(f"Training samples: {len(X_train)} (80%)")
print(f"Testing samples: {len(X_test)} (20%)")

# Create and train Decision Tree Classifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

print(f"\n✓ Model Trained Successfully!")
print(f"Model Type: Decision Tree Classifier")

# Make predictions on test data
y_pred = model.predict(X_test)

print(f"\n✓ Predictions Made!")
print(f"Sample Predictions (First 10): {y_pred[:10]}")
print(f"Actual Labels (First 10): {y_test.values[:10]}")

# ============================================================================
# STEP 4: EVALUATE MODEL'S ACCURACY AND PERFORMANCE
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: MODEL EVALUATION & PERFORMANCE")
print("=" * 80)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✓ Accuracy Score: {accuracy * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\n✓ Confusion Matrix:")
print(cm)

# Classification Report
print(f"\n✓ Classification Report:")
print(classification_report(y_test, y_pred))

# Additional metrics
correct_predictions = np.sum(y_pred == y_test)
wrong_predictions = np.sum(y_pred != y_test)

print(f"\n✓ Detailed Results:")
print(f"  Total Test Samples: {len(y_test)}")
print(f"  Correct Predictions: {correct_predictions}")
print(f"  Wrong Predictions: {wrong_predictions}")
print(f"  Error Rate: {(1 - accuracy) * 100:.2f}%")

# ============================================================================
# STEP 5: VISUALIZATIONS & UNDERSTANDING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: VISUALIZATIONS (Creating Plots...)")
print("=" * 80)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Iris Flower Classification Results', fontsize=16, fontweight='bold')

# Plot 1: Confusion Matrix Heatmap
ax1 = axes[0, 0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1, 
            xticklabels=y.unique(), yticklabels=y.unique())
ax1.set_title('Confusion Matrix', fontweight='bold')
ax1.set_ylabel('Actual')
ax1.set_xlabel('Predicted')

# Plot 2: Accuracy Bar Chart
ax2 = axes[0, 1]
categories = ['Accuracy', 'Error Rate']
values = [accuracy * 100, (1 - accuracy) * 100]
colors = ['green', 'red']
ax2.bar(categories, values, color=colors, alpha=0.7)
ax2.set_ylabel('Percentage (%)')
ax2.set_title('Model Performance', fontweight='bold')
ax2.set_ylim([0, 100])
for i, v in enumerate(values):
    ax2.text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')

# Plot 3: Predictions vs Actual
ax3 = axes[1, 0]
test_samples = len(y_test)
x_pos = np.arange(min(20, test_samples))  # Show first 20
ax3.scatter(x_pos, y_test.values[:20], label='Actual', s=100, alpha=0.6)
ax3.scatter(x_pos, y_pred[:20], label='Predicted', s=100, alpha=0.6)
ax3.set_xlabel('Sample Index')
ax3.set_ylabel('Species (0=setosa, 1=versicolor, 2=virginica)')
ax3.set_title('Predictions vs Actual (First 20 Samples)', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Feature Importance
ax4 = axes[1, 1]
feature_importance = model.feature_importances_
feature_names = X.columns
sorted_idx = np.argsort(feature_importance)[::-1]
ax4.barh(range(len(feature_names)), feature_importance[sorted_idx], alpha=0.7, color='steelblue')
ax4.set_yticks(range(len(feature_names)))
ax4.set_yticklabels([feature_names[i] for i in sorted_idx])
ax4.set_xlabel('Importance')
ax4.set_title('Feature Importance', fontweight='bold')

plt.tight_layout()
plt.savefig(r'C:\Users\ravi\iris_results.png', dpi=100, bbox_inches='tight')
print("\n✓ Plot saved as 'iris_results.png'")
plt.show()

# ============================================================================
# UNDERSTANDING CLASSIFICATION CONCEPTS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: UNDERSTANDING CLASSIFICATION CONCEPTS")
print("=" * 80)

concepts = """
✓ CLASSIFICATION CONCEPTS EXPLAINED:

1. FEATURES (X) - Input Data:
   - Sepal Length, Sepal Width, Petal Length, Petal Width
   - These 4 measurements help identify the species
   
2. TARGET (y) - Output Label:
   - Setosa (0), Versicolor (1), Virginica (2)
   - The species we want to predict

3. TRAINING DATA (80%):
   - Used to teach the model the pattern between features and species
   - Model learns which measurements match which species

4. TEST DATA (20%):
   - Unseen data used to evaluate how well the model generalizes
   - Tests if model can predict correctly on new flowers

5. DECISION TREE:
   - Splits data based on feature values (e.g., if petal_length > 2.45...)
   - Creates a tree of decisions to classify flowers
   - Easy to understand and visualize

6. ACCURACY:
   - Percentage of correct predictions
   - (Correct Predictions / Total Predictions) × 100
   - Our model: {:.2f}%

7. CONFUSION MATRIX:
   - Shows which species are correctly/incorrectly classified
   - Diagonal = correct predictions (✓)
   - Off-diagonal = wrong predictions (✗)

8. OVERFITTING vs UNDERFITTING:
   - Overfitting: Model memorizes training data (bad on test data)
   - Underfitting: Model too simple (bad on both)
   - Goal: Balance = Good on both training and test data
""".format(accuracy * 100)

print(concepts)

# ============================================================================
# MAKE PREDICTIONS ON NEW DATA
# ============================================================================
print("\n" + "=" * 80)
print("BONUS: PREDICT ON NEW FLOWER MEASUREMENTS")
print("=" * 80)

# Example: New flower measurements
new_flower = pd.DataFrame({
    'sepal length': [6.5],
    'sepal width': [3.0],
    'petal length': [5.5],
    'petal width': [1.8]
})

prediction = model.predict(new_flower)
species_map = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}

print(f"\n✓ New Flower Measurements:")
print(new_flower)
print(f"\n✓ Predicted Species: {species_map[prediction[0]]}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("PROJECT SUMMARY")
print("=" * 80)

summary = f"""
✓ Data Loaded: 150 iris flowers with 4 measurements each
✓ Model Trained: Decision Tree Classifier
✓ Accuracy Achieved: {accuracy * 100:.2f}%
✓ Correct Predictions: {correct_predictions} out of {len(y_test)}
✓ Test Samples: {len(X_test)}
✓ Training Samples: {len(X_train)}

CLASSIFICATION SUCCESSFUL! 🎉
"""

print(summary)
print("=" * 80)