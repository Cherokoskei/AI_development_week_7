## Preprocessing data
import pandas as pd
import os

def main():
    df = pd.read_csv("data/raw/student_dropout.csv")
    df.dropna(inplace=True)
    df["dropout"] = (df["Target"] == "Dropout").astype(int)
    df.drop(columns=["ID"], errors="ignore", inplace=True)
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/student_dropout_clean.csv", index=False)
    print("✅ Preprocessed data saved.")

if __name__ == "__main__":
    main()

## Training model

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import os

def main():
    df = pd.read_csv("data/processed/student_dropout_clean.csv")
    X = pd.get_dummies(df.drop(columns=["dropout"]))
    y = df["dropout"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(" Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/random_forest_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    print(" Model and scaler saved to /models.")

if __name__ == "__main__":
    main()

##Evaluating the model

import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv("data/processed/student_dropout_clean.csv")
    X = pd.get_dummies(df.drop(columns=["dropout"]))
    y = df["dropout"]

    scaler = joblib.load("models/scaler.pkl")
    X_scaled = scaler.transform(X)

    model = joblib.load("models/random_forest_model.pkl")
    y_pred = model.predict(X_scaled)

    print(" Evaluation Report:")
    print(classification_report(y, y_pred))
    print("Accuracy:", accuracy_score(y, y_pred))
    print("F1 Score:", f1_score(y, y_pred))

    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Continue", "Dropout"], yticklabels=["Continue", "Dropout"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    main()

