import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

CSV_FILE = "gesture_data.csv"
MODEL_FILE = "gesture.pkl"

data = pd.read_csv(CSV_FILE)

x= data.drop("label", axis=1).values
y= data["label"].values

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

#random forest train
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42
)
clf.fit(x_train, y_train)

#evaluate
y_pred = clf.predict(x_test)
print("\n=== Model Accuracy ===")
print(accuracy_score(y_test, y_pred))

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

# Save the trained model
joblib.dump(clf, MODEL_FILE)
print(f"\nModel saved as {MODEL_FILE}")
