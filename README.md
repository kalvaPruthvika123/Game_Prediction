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
1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
2. Activate the virtual environment:
   ```bash
   venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the main ML script:
   ```bash
   python player_engagement_prediction.py
   ```
5. Run the Streamlit dashboard:
   ```bash
   streamlit run app.py
   ```

## Vercel Deployment
This project also includes a Vercel-ready frontend and API.
- `index.html` is the web UI.
- `api/predict.py` is the Python serverless function for predictions.
- `requirements.txt` defines the Python dependencies.
- `vercel.json` configures the Python runtime.

### Deploy to Vercel
1. Install the Vercel CLI:
   ```bash
   npm install -g vercel
   ```
2. Log in to Vercel:
   ```bash
   vercel login
   ```
3. Deploy the project:
   ```bash
   vercel
   ```

### Run locally with Vercel Dev
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start Vercel local dev server:
   ```bash
   vercel dev
   ```
3. Open this URL in your browser:
   ```text
   http://localhost:3000
   ```

## Dashboard Features
The UI provides a lightweight browser form for player engagement prediction, then shows:
- predicted engagement level
- probability scores for Low / Medium / High

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
