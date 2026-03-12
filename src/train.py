import argparse
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def main(test_size, random_state):
    # Load dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Train model
    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {accuracy:.2f}")
    print("Confusion Matrix:")
    print(cm)

    # Make sure outputs folder exists
    os.makedirs("outputs", exist_ok=True)

    # Save trained model
    joblib.dump(model, "outputs/model.joblib")

    # Save confusion matrix image
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig("outputs/confusion_matrix.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Iris Decision Tree")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)

    args = parser.parse_args()
    main(args.test_size, args.random_state)
    