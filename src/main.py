# Import necessary libraries
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

DATA_PATH = 'data/maternal_health_risk.csv'
LABELS = ['low risk', 'mid risk', 'high risk']

if __name__ == '__main__':
    # Load the Maternal Health Risk dataset
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows")

    # Clean the data: remove duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    print(f"Removed {before - len(df)} duplicate rows")

    # Clean the data: remove physiologically implausible heart rates
    before = len(df)
    df = df[df['HeartRate'] >= 40]
    print(f"Removed {before - len(df)} rows with implausible heart rate")

    print("\nClass distribution after cleaning:")
    print(df['RiskLevel'].value_counts())

    X = df.drop(columns=['RiskLevel'])
    y = df['RiskLevel']

    # Stratified split so all risk levels are represented proportionally
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train a Gradient Boosting classifier
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate on the test set
    y_pred = model.predict(X_test)
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.3f}")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, labels=LABELS))

    print("Confusion matrix (rows = actual, columns = predicted):")
    cm = confusion_matrix(y_test, y_pred, labels=LABELS)
    print(pd.DataFrame(cm, index=LABELS, columns=LABELS))

    # Save the model to the models/ folder (mounted as a volume at runtime)
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/maternal_risk_model.pkl')
    print("Model saved to models/maternal_risk_model.pkl")