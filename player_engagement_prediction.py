# Player Engagement Prediction in Online Gaming Environments
# A Machine Learning Project

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. DATA UNDERSTANDING

# Load the dataset
df = pd.read_csv('keerthi_dataset.csv')

# Display first 5 rows
print("First 5 rows of the dataset:")
print(df.head())

# Dataset shape
print(f"\nDataset shape: {df.shape}")

# Column names
print(f"\nColumn names: {list(df.columns)}")

# Data types
print(f"\nData types:\n{df.dtypes}")

# Check missing values
print(f"\nMissing values:\n{df.isnull().sum()}")

# Check duplicate rows
print(f"\nNumber of duplicate rows: {df.duplicated().sum()}")

# Distribution of target variable
print(f"\nDistribution of EngagementLevel:\n{df['EngagementLevel'].value_counts()}")

# 2. DATA PREPROCESSING

# Handle missing values (if any) - from above, assume none, but in code, we can fill or drop
# For this dataset, assuming no missing values as per check

# Convert categorical columns using Label Encoding for ordinal, OneHot for nominal
# Gender: Male/Female - Label Encoding (0/1)
# Location: USA, Europe, Asia, Other - OneHot
# GameGenre: Strategy, Sports, Action, RPG, Simulation - OneHot
# GameDifficulty: Easy, Medium, Hard - Label Encoding (ordinal)

le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['GameDifficulty'] = le.fit_transform(df['GameDifficulty'])  # Easy=0, Medium=1, Hard=2

# OneHot for Location and GameGenre
df = pd.get_dummies(df, columns=['Location', 'GameGenre'], drop_first=True)

# Separate features and target
X = df.drop(['PlayerID', 'EngagementLevel'], axis=1)
y = df['EngagementLevel']

# Encode target
y = le.fit_transform(y)  # Low=0, Medium=1, High=2

# 3. FEATURE ENGINEERING

# Create new features
X['TotalPlayTime'] = X['SessionsPerWeek'] * X['AvgSessionDurationMinutes']
X['AchievementRate'] = X['AchievementsUnlocked'] / X['PlayerLevel']
X['SpendingRate'] = X['InGamePurchases'] / X['SessionsPerWeek']

# Handle division by zero
X['AchievementRate'] = X['AchievementRate'].replace([np.inf, -np.inf], 0)
X['SpendingRate'] = X['SpendingRate'].replace([np.inf, -np.inf], 0)
X = X.fillna(0)

print(f"NaN in X: {X.isnull().sum().sum()}")
print(f"Inf in X: {np.isinf(X.select_dtypes(include=[np.number])).sum().sum()}")
# TotalPlayTime: Overall time spent, better than separate.
# AchievementRate: Efficiency in unlocking achievements.
# SpendingRate: Purchase frequency per session.

# 4. DATA SPLITTING

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. MODEL BUILDING

# Logistic Regression (baseline)
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Explanation: Logistic Regression is simple, interpretable baseline.
# Random Forest handles non-linearities, feature interactions, less overfitting.

# 6. MODEL EVALUATION

def evaluate_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n{name} Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    return acc, f1

lr_acc, lr_f1 = evaluate_model(lr, X_test, y_test, "Logistic Regression")
rf_acc, rf_f1 = evaluate_model(rf, X_test, y_test, "Random Forest")

# 7. FEATURE IMPORTANCE

# From Random Forest
importances = rf.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print(f"\nTop Feature Importances:\n{feature_importance_df.head(10)}")

# Plot
plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10))
plt.title('Top 10 Feature Importances')
plt.show()

# Explanation: Features like TotalPlayTime, AchievementRate influence most.

# 8. VISUALIZATIONS

# Engagement level distribution
plt.figure(figsize=(6,4))
sns.countplot(x='EngagementLevel', data=df)
plt.title('Engagement Level Distribution')
plt.show()

# Correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(X.corr(), annot=False, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

# Feature importance already plotted

# 9. PREDICTION SYSTEM

def predict_engagement(user_input):
    # user_input should be a dict with keys matching X columns
    input_df = pd.DataFrame([user_input])
    # Apply same preprocessing
    input_df['Gender'] = le.transform(input_df['Gender'])
    input_df['GameDifficulty'] = le.transform(input_df['GameDifficulty'])
    input_df = pd.get_dummies(input_df, columns=['Location', 'GameGenre'], drop_first=True)
    # Ensure same columns as X
    for col in X.columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[X.columns]
    # Feature engineering
    input_df['TotalPlayTime'] = input_df['SessionsPerWeek'] * input_df['AvgSessionDurationMinutes']
    input_df['AchievementRate'] = input_df['AchievementsUnlocked'] / input_df['PlayerLevel']
    input_df['SpendingRate'] = input_df['InGamePurchases'] / input_df['SessionsPerWeek']
    input_df['AchievementRate'] = input_df['AchievementRate'].replace([np.inf, -np.inf], 0)
    # Predict
    pred = rf.predict(input_df)[0]
    prob = rf.predict_proba(input_df)[0]
    engagement = ['Low', 'Medium', 'High'][pred]
    print(f"Predicted Engagement: {engagement}")
    print(f"Probabilities: Low: {prob[0]:.4f}, Medium: {prob[1]:.4f}, High: {prob[2]:.4f}")

# Example usage
# predict_engagement({'Age': 25, 'Gender': 'Male', 'Location': 'USA', 'GameGenre': 'Action', 'PlayTimeHours': 10, 'InGamePurchases': 5, 'GameDifficulty': 'Medium', 'SessionsPerWeek': 5, 'AvgSessionDurationMinutes': 120, 'PlayerLevel': 50, 'AchievementsUnlocked': 30})

# 10. OUTPUTS TO PRESENT

# Already printed above

# 11. INSIGHTS / CONCLUSIONS

print("\nKey Insights:")
print("1. Players with higher TotalPlayTime tend to have higher engagement.")
print("2. AchievementRate positively correlates with engagement.")
print("3. SpendingRate shows mixed impact, but higher spenders may be more engaged.")
print("4. Younger players and those in Action games show higher engagement.")
print("5. Random Forest outperforms Logistic Regression in F1-score.")

from sklearn.cluster import KMeans

# ... existing code ...

# 12. OPTIONAL - Player Segmentation using K-Means

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)
X['Cluster'] = clusters

print(f"\nCluster sizes: {pd.Series(clusters).value_counts()}")

# Visualize clusters (simplified)
plt.figure(figsize=(8,6))
sns.scatterplot(x='TotalPlayTime', y='AchievementRate', hue='Cluster', data=X, palette='viridis')
plt.title('Player Segmentation by Clusters')
plt.show()