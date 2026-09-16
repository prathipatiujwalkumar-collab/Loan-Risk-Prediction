"""
Test script for Predictor.

Author: Ujwal
Project: Intelligent Loan Risk Assessment Platform
"""

from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.predictor import Predictor
from src.config import TARGET_COLUMN


def main():

    # ----------------------------------------
    # Load Dataset
    # ----------------------------------------

    loader = DataLoader()

    df = loader.load_training_data()

    # ----------------------------------------
    # Feature Engineering
    # ----------------------------------------

    engineer = FeatureEngineer()

    df = engineer.transform(df)

    # ----------------------------------------
    # Prepare Sample
    # ----------------------------------------

    X = df.drop(columns=[TARGET_COLUMN])

    sample = X.iloc[[0]]

    # ----------------------------------------
    # Prediction
    # ----------------------------------------

    predictor = Predictor()

    prediction, probability = predictor.predict_single(
        sample
    )

    print("\n" + "=" * 50)
    print("PREDICTION RESULT")
    print("=" * 50)

    print(f"Prediction : {prediction}")

    print(f"Probability: {probability}")

    print("\nPredictor Test Successful")


if __name__ == "__main__":
    main()