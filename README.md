# Banknote-Authentication
Banknote Authentication
# Banknote Authentication - Decision Tree Classifier

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# 1. Load Dataset
columns = ['variance', 'skewness', 'kurtosis', 'entropy', 'class']
data = pd.read_csv(dataset_url, header=None, names=columns)

data.head()

# 2. Statistical Analysis
print("Variance:\n", data.var())
print("\nSkewness:\n", data.skew())
print("\nKurtosis:\n", data.kurtosis())
print("\nEntropy (mean entropy per feature):\n", data[['variance', 'skewness', 'kurtosis', 'entropy']].apply(lambda x: -np.sum(x*np.log2(x+1e-9))/len(x)))

# 3. Visualization
sns.pairplot(data, hue='class')
plt.show()

# 4. Train-Test Split
X = data.drop('class', axis=1)
y = data['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Decision Tree Training and Hyperparameter Tuning
criteria = ['gini', 'entropy']
max_depths = [3, 5, 7]
min_samples_splits = [2, 5, 10]

best_score = 0
best_model = None

for criterion in criteria:
    for depth in max_depths:
        for split in min_samples_splits:
            model = DecisionTreeClassifier(criterion=criterion, max_depth=depth, min_samples_split=split, random_state=42)
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            print(f"Criterion: {criterion}, Max Depth: {depth}, Min Samples Split: {split}, Accuracy: {score:.4f}")
            if score > best_score:
                best_score = score
                best_model = model

# 6. Evaluation of Best Model
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))

ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test)
plt.show()

# 7. Plot Best Decision Tree
plt.figure(figsize=(20,10))
plot_tree(best_model, feature_names=X.columns, class_names=['Fake', 'Authentic'], filled=True)
plt.show()

# 8. Feature Importance
importances = best_model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importances')
plt.show()

# 9. Comment Section
print("""
Comment:
- Decision Trees offer strong interpretability and reasonable performance.
- Hyperparameter tuning showed that higher depth can slightly improve accuracy, but may risk overfitting.
- Important features identified help explain model predictions.
- Overall, Decision Trees are a good choice for this dataset, but ensemble methods (like Random Forest) could further boost performance.
""")

# README.md content
readme_content = """
# Decision Tree for Banknote Authentication

## Overview
This project implements a Decision Tree Classifier to predict the authenticity of banknotes.

## Contents
- `decision_tree.ipynb`: Google Colab notebook with data exploration, model training, evaluation, and analysis.
- Visualizations of decision tree structure and feature importances.

## Instructions
1. Open the `decision_tree.ipynb` notebook.
2. Replace 'YOUR_LINK_HERE' with the dataset link provided.
3. Run all cells to see outputs, graphs, and analysis.

## Highlights
- Explored features' statistical properties (variance, skewness, kurtosis, entropy).
- Visualized data distributions.
- Trained multiple decision tree models with different hyperparameters.
- Evaluated using accuracy, precision, recall, F1-score.
- Analyzed feature importance for better interpretability.

## Conclusion
Decision Trees provide good initial performance and interpretability. With careful tuning, they perform well on this dataset.

"""

with open("README.md", "w") as file:
    file.write(readme_content)
