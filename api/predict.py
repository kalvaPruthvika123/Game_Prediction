import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

LABELS = ['Low', 'Medium', 'High']


def load_model():
    df = pd.read_csv('keerthi_dataset.csv')
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
    X = X.replace([np.inf, -np.inf], 0).fillna(0)

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model, gender_le, difficulty_le, target_le, X.columns.tolist()

MODEL, GENDER_LE, DIFFICULTY_LE, TARGET_LE, FEATURE_COLUMNS = load_model()


def preprocess_input(body):
    data = {
        'Age': int(body.get('Age', 25)),
        'Gender': int(GENDER_LE.transform([body.get('Gender', 'Male')])[0]),
        'PlayTimeHours': float(body.get('PlayTimeHours', 10.0)),
        'InGamePurchases': int(body.get('InGamePurchases', 1)),
        'GameDifficulty': int(DIFFICULTY_LE.transform([body.get('GameDifficulty', 'Medium')])[0]),
        'SessionsPerWeek': int(body.get('SessionsPerWeek', 5)),
        'AvgSessionDurationMinutes': float(body.get('AvgSessionDurationMinutes', 120.0)),
        'PlayerLevel': int(body.get('PlayerLevel', 50)),
        'AchievementsUnlocked': int(body.get('AchievementsUnlocked', 25)),
        'Location_Europe': 1 if body.get('Location') == 'Europe' else 0,
        'Location_Other': 1 if body.get('Location') == 'Other' else 0,
        'Location_USA': 1 if body.get('Location') == 'USA' else 0,
        'GameGenre_RPG': 1 if body.get('GameGenre') == 'RPG' else 0,
        'GameGenre_Simulation': 1 if body.get('GameGenre') == 'Simulation' else 0,
        'GameGenre_Sports': 1 if body.get('GameGenre') == 'Sports' else 0,
        'GameGenre_Strategy': 1 if body.get('GameGenre') == 'Strategy' else 0,
    }

    input_df = pd.DataFrame([data])
    for col in FEATURE_COLUMNS:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[FEATURE_COLUMNS]

    input_df['TotalPlayTime'] = input_df['SessionsPerWeek'] * input_df['AvgSessionDurationMinutes']
    input_df['AchievementRate'] = input_df['AchievementsUnlocked'] / input_df['PlayerLevel']
    input_df['SpendingRate'] = input_df['InGamePurchases'] / input_df['SessionsPerWeek']
    input_df = input_df.replace([np.inf, -np.inf], 0).fillna(0)

    return input_df


def handler(request):
    try:
        body = request.get_json(force=True)
    except Exception:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'Invalid JSON payload'}),
            'headers': {'Content-Type': 'application/json'}
        }

    input_df = preprocess_input(body)
    prediction = MODEL.predict(input_df)[0]
    probabilities = MODEL.predict_proba(input_df)[0].tolist()

    return {
        'statusCode': 200,
        'body': json.dumps({
            'prediction': LABELS[prediction],
            'probabilities': {
                'Low': probabilities[0],
                'Medium': probabilities[1],
                'High': probabilities[2]
            }
        }),
        'headers': {
            'Content-Type': 'application/json'
        }
    }
