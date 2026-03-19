import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib
import os

def train_model():
    print("Loading dataset...")
    file_path = '../data/air_quality.csv'
    
    if not os.path.exists(file_path):
        print("Error: Please download the Kaggle dataset and save it as 'air_quality.csv' inside the 'data/' folder.")
        return

    df = pd.read_csv(file_path)

    # Exact columns based on the Kaggle dataset
    features = [
        'Temperature', 'Humidity', 'PM2.5', 'PM10', 
        'NO2', 'SO2', 'CO', 'Proximity_to_Industrial_Areas', 'Population_Density'
    ]
    target = 'Air Quality'

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Building and training the pipeline...")
    # Pipeline handles scaling and classification in one continuous sequence
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])

    pipeline.fit(X_train, y_train)
    
    accuracy = pipeline.score(X_test, y_test)
    print(f"Model trained successfully! Test Accuracy: {accuracy * 100:.2f}%")

    # Save the compiled pipeline
    joblib.dump(pipeline, '../models/aqi_pipeline.pkl')
    print("Pipeline saved to 'models/aqi_pipeline.pkl'")

if __name__ == "__main__":
    train_model()