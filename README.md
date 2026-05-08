# Player Engagement Prediction in Online Gaming Environments

This is a complete Machine Learning project to predict player engagement levels (Low, Medium, High) based on gaming data.

## Dataset
- **File**: keerthi_dataset.csv
- **Target**: EngagementLevel
- **Features**: Age, Gender, Location, GameGenre, PlayTimeHours, InGamePurchases, GameDifficulty, SessionsPerWeek, AvgSessionDurationMinutes, PlayerLevel, AchievementsUnlocked

## Project Structure
- `player_engagement_prediction.py`: Main script with all steps
- `app.py`: Streamlit web app for prediction
- `keerthi_dataset.csv`: Dataset

## How to Run
1. Install dependencies: `pip install pandas numpy scikit-learn matplotlib seaborn streamlit`
2. Run the main script: `python player_engagement_prediction.py`
3. Run the app: `streamlit run app.py`

## Steps Covered
1. Data Understanding
2. Data Preprocessing
3. Feature Engineering
4. Data Splitting
5. Model Building (Logistic Regression & Random Forest)
6. Model Evaluation
7. Feature Importance
8. Visualizations
9. Prediction System
10. Insights

## Key Findings
- Random Forest achieves 92% accuracy
- TotalPlayTime is the most important feature
- Higher playtime and achievements correlate with higher engagement