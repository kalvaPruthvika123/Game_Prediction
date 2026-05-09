import os

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Set page config
st.set_page_config(page_title="Player Engagement Prediction", layout="wide")

# Load and preprocess data
@st.cache_data
def load_data():
    dataset_path = os.path.join(os.path.dirname(__file__), 'keerthi_dataset.csv')
    if not os.path.exists(dataset_path):
        st.error(f"Dataset file not found: {dataset_path}")
        st.stop()

    df = pd.read_csv(dataset_path)
    gender_le = LabelEncoder()
    difficulty_le = LabelEncoder()
    target_le = LabelEncoder()

    df['Gender'] = gender_le.fit_transform(df['Gender'])
    df['GameDifficulty'] = difficulty_le.fit_transform(df['GameDifficulty'])
    df = pd.get_dummies(df, columns=['Location', 'GameGenre'], drop_first=True)
    X = df.drop(['PlayerID', 'EngagementLevel'], axis=1)
    y = target_le.fit_transform(df['EngagementLevel'])
    X['TotalPlayTime'] = X['SessionsPerWeek'] * X['AvgSessionDurationMinutes']
    X['AchievementRate'] = X['AchievementsUnlocked'] / X['PlayerLevel']
    X['SpendingRate'] = X['InGamePurchases'] / X['SessionsPerWeek']
    X['AchievementRate'] = X['AchievementRate'].replace([np.inf, -np.inf], 0)
    X['SpendingRate'] = X['SpendingRate'].replace([np.inf, -np.inf], 0)
    X = X.fillna(0)
    return df, X, y, gender_le, difficulty_le, target_le

df, X, y, gender_le, difficulty_le, target_le = load_data()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
@st.cache_resource
def train_model():
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    return rf

rf = train_model()

# Compute additional data
@st.cache_data
def compute_metrics():
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    return acc, prec, rec, f1, cm

acc, prec, rec, f1, cm = compute_metrics()

@st.cache_data
def compute_feature_importance():
    importances = rf.feature_importances_
    feature_names = X.columns
    feature_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_df = feature_df.sort_values(by='Importance', ascending=False)
    return feature_df

feature_df = compute_feature_importance()

@st.cache_data
def compute_clusters():
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X)
    X_clustered = X.copy()
    X_clustered['Cluster'] = clusters
    return X_clustered

X_clustered = compute_clusters()

# Title
st.title("Player Engagement Prediction Dashboard")

# Sidebar for navigation
option = st.sidebar.selectbox(
    "Choose what to view:",
    ["Prediction Tool", "Data Overview", "Model Performance", "Feature Importance", "Visualizations", "Player Segmentation"]
)

if option == "Prediction Tool":
    st.header("Predict Player Engagement")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Input Features")
        age = st.slider("Age", 18, 60, 25)
        gender = st.selectbox("Gender", ["Male", "Female"])
        location = st.selectbox("Location", ["USA", "Europe", "Asia", "Other"])
        game_genre = st.selectbox("Game Genre", ["Strategy", "Sports", "Action", "RPG", "Simulation"])
        play_time = st.slider("Play Time Hours", 0.0, 50.0, 10.0)
        purchases = st.slider("In-Game Purchases", 0, 10, 1)
        difficulty = st.selectbox("Game Difficulty", ["Easy", "Medium", "Hard"])
        sessions = st.slider("Sessions Per Week", 0, 20, 5)
        avg_duration = st.slider("Avg Session Duration Minutes", 30, 200, 120)
        level = st.slider("Player Level", 1, 100, 50)
        achievements = st.slider("Achievements Unlocked", 0, 100, 25)

    # Preprocess input
    input_data = {
        'Age': age,
        'Gender': int(gender_le.transform([gender])[0]),
        'PlayTimeHours': play_time,
        'InGamePurchases': purchases,
        'GameDifficulty': int(difficulty_le.transform([difficulty])[0]),
        'SessionsPerWeek': sessions,
        'AvgSessionDurationMinutes': avg_duration,
        'PlayerLevel': level,
        'AchievementsUnlocked': achievements,
        'Location_Europe': 1 if location == 'Europe' else 0,
        'Location_Other': 1 if location == 'Other' else 0,
        'Location_USA': 1 if location == 'USA' else 0,
        'GameGenre_RPG': 1 if game_genre == 'RPG' else 0,
        'GameGenre_Simulation': 1 if game_genre == 'Simulation' else 0,
        'GameGenre_Sports': 1 if game_genre == 'Sports' else 0,
        'GameGenre_Strategy': 1 if game_genre == 'Strategy' else 0,
    }

    input_df = pd.DataFrame([input_data])
    # Ensure all columns
    for col in X.columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[X.columns]

    # Feature engineering
    input_df['TotalPlayTime'] = input_df['SessionsPerWeek'] * input_df['AvgSessionDurationMinutes']
    input_df['AchievementRate'] = input_df['AchievementsUnlocked'] / input_df['PlayerLevel']
    input_df['SpendingRate'] = input_df['InGamePurchases'] / input_df['SessionsPerWeek']
    input_df['AchievementRate'] = input_df['AchievementRate'].replace([np.inf, -np.inf], 0)
    input_df['SpendingRate'] = input_df['SpendingRate'].replace([np.inf, -np.inf], 0)
    input_df = input_df.fillna(0)

    with col2:
        st.subheader("Prediction Result")
        if st.button("Predict Engagement"):
            pred = rf.predict(input_df)[0]
            prob = rf.predict_proba(input_df)[0]
            engagement_label = target_le.inverse_transform([pred])[0]
            probability_labels = target_le.classes_
            st.success(f"Predicted Engagement Level: **{engagement_label}**")
            st.write("### Probabilities:")
            for label, value in zip(probability_labels, prob):
                st.write(f"{label}: {value:.4f}")

elif option == "Data Overview":
    st.header("Data Overview")

    st.subheader("Dataset Shape")
    st.write(f"Number of rows: {df.shape[0]}, Number of columns: {df.shape[1]}")

    st.subheader("First 5 Rows")
    st.dataframe(df.head())

    st.subheader("Data Types")
    st.write(df.dtypes)

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    st.subheader("Engagement Level Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='EngagementLevel', data=df, ax=ax)
    st.pyplot(fig)

    st.subheader("Summary Statistics")
    st.dataframe(df.describe())

elif option == "Model Performance":
    st.header("Model Performance")

    st.subheader("Random Forest Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{acc:.4f}")
    col2.metric("Precision", f"{prec:.4f}")
    col3.metric("Recall", f"{rec:.4f}")
    col4.metric("F1-Score", f"{f1:.4f}")

    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)

elif option == "Feature Importance":
    st.header("Feature Importance")

    st.subheader("Top 10 Features")
    st.dataframe(feature_df.head(10))

    st.subheader("Feature Importance Plot")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.barplot(x='Importance', y='Feature', data=feature_df.head(10), ax=ax)
    st.pyplot(fig)

elif option == "Visualizations":
    st.header("Visualizations")

    st.subheader("Feature Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12,8))
    sns.heatmap(X.corr(), annot=False, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.subheader("Engagement vs Total Play Time")
    fig, ax = plt.subplots()
    sns.boxplot(x='EngagementLevel', y='TotalPlayTime', data=df.assign(EngagementLevel=target_le.inverse_transform(y)), ax=ax)
    st.pyplot(fig)

elif option == "Player Segmentation":
    st.header("Player Segmentation")

    st.subheader("Cluster Sizes")
    cluster_counts = X_clustered['Cluster'].value_counts().sort_index()
    st.bar_chart(cluster_counts)

    st.subheader("Clusters Visualization")
    fig, ax = plt.subplots(figsize=(8,6))
    sns.scatterplot(x='TotalPlayTime', y='AchievementRate', hue='Cluster', data=X_clustered, palette='viridis', ax=ax)
    st.pyplot(fig)

    st.subheader("Cluster Characteristics")
    for cluster in range(3):
        st.write(f"**Cluster {cluster}:**")
        cluster_data = X_clustered[X_clustered['Cluster'] == cluster]
        st.write(cluster_data[['TotalPlayTime', 'AchievementRate', 'SpendingRate']].describe())