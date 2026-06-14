import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 80)
print("IRIS FLOWER CLASSIFICATION - COMPLETE IMPROVED PROJECT")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD AND EXPLORE DATA
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: LOAD AND EXPLORE DATA")
print("=" * 80)

# Load CSV file
df = pd.read_csv(r'C:\Users\ravi\Downloads\iris.csv')

print("\n✓ Dataset Loaded Successfully!")
print(f"Total Samples: {len(df)}")
print(f"Column Names: {df.columns.tolist()}")

# Extract measurements (features) - X
X = df.iloc[:, :-1]
print(f"\nFeatures (Measurements) Shape: {X.shape}")
print(f"Feature Names: {X.columns.tolist()}")

# Extract species labels (target) - y
y = df.iloc[:, -1]
print(f"\nTarget (Species) Shape: {y.shape}")
print(f"Species Names: {y.unique()}")

# Auto detect column names
col0 = X.columns[0]  # sepal length
col1 = X.columns[1]  # sepal width
col2 = X.columns[2]  # petal length
col3 = X.columns[3]  # petal width
print(f"\n✓ Columns Detected:")
print(f"  Col 0: {col0}")
print(f"  Col 1: {col1}")
print(f"  Col 2: {col2}")
print(f"  Col 3: {col3}")

# Show sample data
print("\nSample Data (First 5 rows):")
print(df.head())

# Show statistics
print("\nMeasurements Statistics:")
print(X.describe())

# ============================================================================
# STEP 2: TRAIN TEST SPLIT
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: TRAIN TEST SPLIT (80% Train, 20% Test)")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n✓ Data Split Done!")
print(f"Training samples: {len(X_train)} (80%)")
print(f"Testing samples:  {len(X_test)} (20%)")

# ============================================================================
# STEP 3: TRAIN ALL MODELS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: TRAIN ALL MODELS")
print("=" * 80)

# Define all models
models = {
    'Decision Tree':  DecisionTreeClassifier(random_state=42),
    'KNN':            KNeighborsClassifier(),
    'SVM':            SVC(random_state=42),
    'Random Forest':  RandomForestClassifier(random_state=42)
}

# Train all models and store results
model_accuracies = {}
model_cv_scores = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    model_accuracies[name] = accuracy * 100
    cv_scores = cross_val_score(model, X, y, cv=5)
    model_cv_scores[name] = cv_scores
    print(f"\n✓ {name}:")
    print(f"   Test Accuracy:      {accuracy * 100:.2f}%")
    print(f"   Cross Val Accuracy: {cv_scores.mean() * 100:.2f}%")
    print(f"   Cross Val Std:      {cv_scores.std() * 100:.2f}%")

# Best model
best_model_name = max(model_accuracies, key=model_accuracies.get)
best_model = models[best_model_name]
print(f"\n✓ Best Model: {best_model_name} ({model_accuracies[best_model_name]:.2f}%)")

# ============================================================================
# STEP 4: EVALUATE BEST MODEL
# ============================================================================
print("\n" + "=" * 80)
print(f"STEP 4: EVALUATE BEST MODEL ({best_model_name})")
print("=" * 80)

y_pred_best = best_model.predict(X_test)
best_accuracy = accuracy_score(y_test, y_pred_best)
cm = confusion_matrix(y_test, y_pred_best)

print(f"\n✓ Accuracy Score:  {best_accuracy * 100:.2f}%")
print(f"\n✓ Confusion Matrix:")
print(cm)
print(f"\n✓ Classification Report:")
print(classification_report(y_test, y_pred_best))

correct_predictions = np.sum(y_pred_best == y_test)
wrong_predictions = np.sum(y_pred_best != y_test)

print(f"\n✓ Detailed Results:")
print(f"  Total Test Samples:   {len(y_test)}")
print(f"  Correct Predictions:  {correct_predictions}")
print(f"  Wrong Predictions:    {wrong_predictions}")
print(f"  Error Rate:           {(1 - best_accuracy) * 100:.2f}%")

# ============================================================================
# STEP 5: ALL 8 VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: CREATING ALL 8 VISUALIZATIONS")
print("=" * 80)

# ─────────────────────────────────────────────
# FIGURE 1: First 4 Plots
# ─────────────────────────────────────────────
fig1, axes1 = plt.subplots(2, 2, figsize=(16, 12))
fig1.suptitle('Iris Classification - Part 1 (Core Results)',
              fontsize=18, fontweight='bold', color='#2C3E50')

# ── Plot 1: Confusion Matrix ──
ax1 = axes1[0, 0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
            xticklabels=y.unique(), yticklabels=y.unique(),
            linewidths=2, linecolor='white')
ax1.set_title('1. Confusion Matrix', fontweight='bold', fontsize=13)
ax1.set_ylabel('Actual Species', fontweight='bold')
ax1.set_xlabel('Predicted Species', fontweight='bold')


# ── Plot 2: Accuracy Bar Chart ──
ax2 = axes1[0, 1]
categories = ['Accuracy', 'Error Rate']
values = [best_accuracy * 100, (1 - best_accuracy) * 100]
colors = ['#2ECC71', '#E74C3C']
ax2.bar(categories, values, color=colors, alpha=0.85,
        edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Percentage (%)', fontweight='bold')
ax2.set_title('2. Accuracy Bar Chart', fontweight='bold', fontsize=13)
ax2.set_ylim([0, 110])
for i, v in enumerate(values):
    ax2.text(i, v + 2, f'{v:.1f}%', ha='center',
             fontweight='bold', fontsize=12)
ax2.grid(axis='y', alpha=0.3)

# ── Plot 3: Predictions vs Actual ──
ax3 = axes1[1, 0]
x_pos = np.arange(min(20, len(y_test)))
ax3.scatter(x_pos, y_test.values[:20], label='Actual',
            s=120, alpha=0.8, color='#3498DB',
            edgecolors='black', linewidth=1.5, marker='o')
ax3.scatter(x_pos, y_pred_best[:20], label='Predicted',
            s=120, alpha=0.8, color='#E67E22',
            edgecolors='black', linewidth=1.5, marker='^')
ax3.set_xlabel('Sample Index', fontweight='bold')
ax3.set_ylabel('Species', fontweight='bold')
ax3.set_title('3. Predictions vs Actual (First 20)',
              fontweight='bold', fontsize=13)
ax3.legend()
ax3.grid(True, alpha=0.3)

# ── Plot 4: Feature Importance ──
ax4 = axes1[1, 1]
dt_model = models['Decision Tree']
feature_importance = dt_model.feature_importances_
feature_names = X.columns
sorted_idx = np.argsort(feature_importance)[::-1]
colors_feat = ['#9B59B6', '#E74C3C', '#F39C12', '#1ABC9C']
ax4.barh(range(len(feature_names)),
         feature_importance[sorted_idx],
         alpha=0.85,
         color=[colors_feat[i] for i in range(len(feature_names))],
         edgecolor='black', linewidth=1.5)
ax4.set_yticks(range(len(feature_names)))
ax4.set_yticklabels([feature_names[i] for i in sorted_idx],
                    fontweight='bold')
ax4.set_xlabel('Importance Score', fontweight='bold')
ax4.set_title('4. Feature Importance', fontweight='bold', fontsize=13)
ax4.grid(axis='x', alpha=0.3)
for i, (idx, v) in enumerate(
        zip(sorted_idx, feature_importance[sorted_idx])):
    ax4.text(v + 0.01, i, f'{v:.3f}', va='center',
             fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig(
    r'C:\Users\ravi\OneDrive\Desktop\python\iris_results_part1.png',
    dpi=150, bbox_inches='tight')
print("\n✓ Part 1 plots saved!")
plt.show(block=False)

# ─────────────────────────────────────────────
# FIGURE 2: Next 4 Plots
# ─────────────────────────────────────────────
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
fig2.suptitle('Iris Classification - Part 2 (Advanced Results)',
              fontsize=18, fontweight='bold', color='#2C3E50')

# ── Plot 5: Model Comparison Chart ──
ax5 = axes2[0, 0]
model_names = list(model_accuracies.keys())
accuracies = list(model_accuracies.values())
colors_models = ['#3498DB', '#E74C3C', '#2ECC71', '#9B59B6']
ax5.bar(model_names, accuracies,
        color=colors_models, alpha=0.85,
        edgecolor='black', linewidth=1.5)
ax5.set_ylabel('Accuracy (%)', fontweight='bold')
ax5.set_title('5. Model Comparison Chart',
              fontweight='bold', fontsize=13)
ax5.set_ylim([80, 105])
ax5.grid(axis='y', alpha=0.3)
for i, v in enumerate(accuracies):
    ax5.text(i, v + 0.5, f'{v:.1f}%', ha='center',
             fontweight='bold', fontsize=11)
ax5.tick_params(axis='x', rotation=15)

# ── Plot 6: Pair Plot - Auto Detect Column Names ──
ax6 = axes2[0, 1]
colors_species = ['#FF6B6B', '#4ECDC4', '#45B7D1']
X_reset = X.reset_index(drop=True)
y_reset = y.reset_index(drop=True)
for i, species in enumerate(y_reset.unique()):
    mask = y_reset == species
    ax6.scatter(
        X_reset[col0][mask],
        X_reset[col1][mask],
        label=str(species),
        alpha=0.7, s=80,
        color=colors_species[i],
        edgecolors='black',
        linewidth=0.5)
ax6.set_xlabel(col0, fontweight='bold')
ax6.set_ylabel(col1, fontweight='bold')
ax6.set_title('6. Pair Plot - Sepal Features',
              fontweight='bold', fontsize=13)
ax6.legend(title='Species', fontsize=9)
ax6.grid(True, alpha=0.3)

# ── Plot 7: Species Distribution ──
ax7 = axes2[1, 0]
species_counts = y.value_counts()
colors_dist = ['#FF6B6B', '#4ECDC4', '#45B7D1']
ax7.pie(
    species_counts,
    labels=species_counts.index,
    autopct='%1.1f%%',
    colors=colors_dist,
    startangle=90,
    explode=(0.05, 0.05, 0.05),
    textprops={'fontweight': 'bold', 'fontsize': 11})
ax7.set_title('7. Species Distribution',
              fontweight='bold', fontsize=13)

# ── Plot 8: Cross Validation Scores ──
ax8 = axes2[1, 1]
cv_means = [model_cv_scores[name].mean() * 100 for name in model_names]
cv_stds  = [model_cv_scores[name].std()  * 100 for name in model_names]
colors_cv = ['#3498DB', '#E74C3C', '#2ECC71', '#9B59B6']
ax8.bar(model_names, cv_means,
        yerr=cv_stds, color=colors_cv,
        alpha=0.85, edgecolor='black',
        linewidth=1.5, capsize=8)
ax8.set_ylabel('CV Accuracy (%)', fontweight='bold')
ax8.set_title('8. Cross Validation Scores (5-Fold)',
              fontweight='bold', fontsize=13)
ax8.set_ylim([80, 110])
ax8.grid(axis='y', alpha=0.3)
for i, (v, std) in enumerate(zip(cv_means, cv_stds)):
    ax8.text(i, v + std + 1, f'{v:.1f}%',
             ha='center', fontweight='bold', fontsize=10)
ax8.tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig(
    r'C:\Users\ravi\OneDrive\Desktop\python\iris_results_part2.png',
    dpi=150, bbox_inches='tight')
print("✓ Part 2 plots saved!")
plt.show()

# ============================================================================
# STEP 6: UNDERSTANDING CLASSIFICATION CONCEPTS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: UNDERSTANDING CLASSIFICATION CONCEPTS")
print("=" * 80)

concepts = f"""
✓ CLASSIFICATION CONCEPTS EXPLAINED:

1. FEATURES (X) - Input Data:
   - {col0}, {col1}, {col2}, {col3}
   - These 4 measurements help identify the species

2. TARGET (y) - Output Label:
   - Species we want to predict

3. TRAINING DATA (80%):
   - {len(X_train)} samples used to teach the model

4. TEST DATA (20%):
   - {len(X_test)} samples used to evaluate performance

5. DECISION TREE:
   - Splits data based on feature values
   - Easy to understand and visualize

6. KNN (K-Nearest Neighbors):
   - Finds K closest data points
   - Classifies based on majority vote

7. SVM (Support Vector Machine):
   - Finds best boundary between classes
   - Works well for small datasets

8. RANDOM FOREST:
   - Collection of multiple Decision Trees
   - More accurate than single tree

9. ACCURACY:
   - (Correct Predictions / Total) x 100
   - Best Model: {best_model_name} = {best_accuracy * 100:.2f}%

10. CONFUSION MATRIX:
    - Diagonal = correct predictions
    - Off-diagonal = wrong predictions

11. CROSS VALIDATION (5-Fold):
    - Splits data into 5 parts
    - Tests model 5 times
    - More reliable than single test

12. OVERFITTING vs UNDERFITTING:
    - Overfitting: Memorizes training data
    - Underfitting: Too simple
    - Goal: Balance = Good on both
"""
print(concepts)

# ============================================================================
# STEP 7: PREDICT ON NEW FLOWER
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: PREDICT ON NEW FLOWER MEASUREMENTS")
print("=" * 80)

new_flower = pd.DataFrame({
    col0: [6.5],
    col1: [3.0],
    col2: [5.5],
    col3: [1.8]
})

print(f"\n✓ New Flower Measurements:")
print(new_flower)

print(f"\n✓ Predictions from ALL Models:")
for name, model in models.items():
    pred = model.predict(new_flower)[0]
    print(f"   {name:20s}: {pred}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("FINAL PROJECT SUMMARY")
print("=" * 80)

print(f"""
✓ Dataset:          150 iris flowers, 4 measurements each
✓ Train Samples:    {len(X_train)} (80%)
✓ Test Samples:     {len(X_test)} (20%)

✓ MODEL RESULTS:
""")

for name in model_names:
    cv_mean = model_cv_scores[name].mean() * 100
    print(f"   {name:20s}: Test={model_accuracies[name]:.2f}%  CV={cv_mean:.2f}%")

print(f"""
✓ Best Model:       {best_model_name} ({best_accuracy * 100:.2f}%)
✓ Correct:          {correct_predictions} out of {len(y_test)}
✓ Wrong:            {wrong_predictions}

✓ PLOTS SAVED:
   iris_results_part1.png  (Plots 1-4)
   iris_results_part2.png  (Plots 5-8)

CLASSIFICATION SUCCESSFUL! 🎉
""")
print("=" * 80)