import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load and preprocess data (same as before)
@st.cache_data
def load_data():
    df = pd.read_csv('keerthi_dataset.csv')
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])
    df['GameDifficulty'] = le.fit_transform(df['GameDifficulty'])
    df = pd.get_dummies(df, columns=['Location', 'GameGenre'], drop_first=True)
    X = df.drop(['PlayerID', 'EngagementLevel'], axis=1)
    y = le.fit_transform(df['EngagementLevel'])
    X['TotalPlayTime'] = X['SessionsPerWeek'] * X['AvgSessionDurationMinutes']
    X['AchievementRate'] = X['AchievementsUnlocked'] / X['PlayerLevel']
    X['SpendingRate'] = X['InGamePurchases'] / X['SessionsPerWeek']
    X['AchievementRate'] = X['AchievementRate'].replace([np.inf, -np.inf], 0)
    X['SpendingRate'] = X['SpendingRate'].replace([np.inf, -np.inf], 0)
    X = X.fillna(0)
    return X, y, le

X, y, le = load_data()

# Train model
@st.cache_resource
def train_model():
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X, y)
    return rf

rf = train_model()

st.title("Player Engagement Prediction")

st.sidebar.header("User Input Features")

age = st.sidebar.slider("Age", 18, 60, 25)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
location = st.sidebar.selectbox("Location", ["USA", "Europe", "Asia", "Other"])
game_genre = st.sidebar.selectbox("Game Genre", ["Strategy", "Sports", "Action", "RPG", "Simulation"])
play_time = st.sidebar.slider("Play Time Hours", 0.0, 50.0, 10.0)
purchases = st.sidebar.slider("In-Game Purchases", 0, 10, 1)
difficulty = st.sidebar.selectbox("Game Difficulty", ["Easy", "Medium", "Hard"])
sessions = st.sidebar.slider("Sessions Per Week", 0, 20, 5)
avg_duration = st.sidebar.slider("Avg Session Duration Minutes", 30, 200, 120)
level = st.sidebar.slider("Player Level", 1, 100, 50)
achievements = st.sidebar.slider("Achievements Unlocked", 0, 100, 25)

# Preprocess input
input_data = {
    'Age': age,
    'Gender': gender,
    'PlayTimeHours': play_time,
    'InGamePurchases': purchases,
    'GameDifficulty': difficulty,
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

if st.button("Predict Engagement"):
    pred = rf.predict(input_df)[0]
    prob = rf.predict_proba(input_df)[0]
    engagement = ['Low', 'Medium', 'High'][pred]
    st.success(f"Predicted Engagement Level: {engagement}")
    st.write(f"Probabilities:")
    st.write(f"Low: {prob[0]:.4f}")
    st.write(f"Medium: {prob[1]:.4f}")
    st.write(f"High: {prob[2]:.4f}")