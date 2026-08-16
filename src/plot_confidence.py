import json
import matplotlib.pyplot as plt
import numpy as np

HISTORY_PATH = "results/prediction_history.json"


# Load prediction history
with open(HISTORY_PATH, "r") as file:
    history = json.load(file)


if not history:
    print("No prediction history found.")
    exit()


# Extract data
predictions = np.arange(1, len(history) + 1)
confidence = [item["confidence"] for item in history]

# Convert result names
results = [item["result"].capitalize() for item in history]


# Create smooth-looking curve
x_smooth = np.linspace(
    predictions.min(),
    predictions.max(),
    200
)

if len(predictions) > 1:
    y_smooth = np.interp(
        x_smooth,
        predictions,
        confidence
    )
else:
    x_smooth = predictions
    y_smooth = confidence


# Create graph
plt.figure(figsize=(10, 6))

# Filled area
plt.fill_between(
    x_smooth,
    y_smooth,
    alpha=0.15
)

# Confidence curve
plt.plot(
    x_smooth,
    y_smooth,
    linewidth=3,
    label="Confidence"
)

# Actual prediction points
plt.scatter(
    predictions,
    confidence,
    s=70,
    zorder=3
)


# Add labels to actual points
for x, y, result in zip(predictions, confidence, results):

    plt.annotate(
        f"{result}\n{y:.2f}%",
        (x, y),
        xytext=(0, 12),
        textcoords="offset points",
        ha="center"
    )


# Graph formatting
plt.title("Leaf Prediction Confidence")
plt.xlabel("Prediction Number")
plt.ylabel("Confidence (%)")

plt.ylim(0, 100)

if len(predictions) > 1:
    plt.xticks(predictions)

plt.grid(
    True,
    alpha=0.25
)

plt.legend()

plt.tight_layout()

plt.show()