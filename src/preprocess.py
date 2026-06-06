from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load
X_train_raw, y_train_raw, X_test_raw, y_test_raw = load_emnist()

# Preprocess
X_train, y_train = preprocess(X_train_raw, y_train_raw)
X_test, y_test = preprocess(X_test_raw, y_test_raw)

# Flatten images
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

print(X_train.shape)
print(X_test.shape)

# Train
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Predict
preds = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, preds)
print("Accuracy:", acc)