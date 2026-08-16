import json
import os
import matplotlib.pyplot as plt


HISTORY_FILE = "results/prediction_history.json"


def load_history():

    if not os.path.exists(HISTORY_FILE):
        print("No prediction history found.")
        return []

    with open(HISTORY_FILE, "r") as file:
        return json.load(file)


def plot_prediction_counts(history):

    healthy_count = 0
    diseased_count = 0

    for prediction in history:

        if prediction["result"] == "healthy":
            healthy_count += 1

        elif prediction["result"] == "diseased":
            diseased_count += 1

    labels = ["Healthy", "Diseased"]
    counts = [healthy_count, diseased_count]

    plt.figure(figsize=(7, 5))

    plt.bar(labels, counts)

    plt.title("Leaf Prediction Results")
    plt.xlabel("Prediction")
    plt.ylabel("Number of Predictions")

    plt.show()


if __name__ == "__main__":

    history = load_history()

    if history:
        plot_prediction_counts(history)
    else:
        print("No predictions available.")