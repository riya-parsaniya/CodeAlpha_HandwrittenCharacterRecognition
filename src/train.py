import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def train_model(X_train, y_train, X_val, y_val):
    # Flatten images: (N, 28, 28) -> (N, 784)
    X_train = X_train.reshape(len(X_train), -1)
    X_val = X_val.reshape(len(X_val), -1)

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    print("Training model...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    print("Accuracy:", accuracy_score(y_val, y_pred))
    print(classification_report(y_val, y_pred))

    return model

def save_model(model, name):
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    joblib.dump(model, path)
    print(f"Model saved: {path}")

def load_model(name):
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    return joblib.load(path)