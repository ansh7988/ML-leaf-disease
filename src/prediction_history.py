import json
from datetime import datetime
from pathlib import Path


# ==================================================
# HISTORY FILE
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

HISTORY_FILE = BASE_DIR / "results" / "prediction_history.json"


# Make sure results folder exists
HISTORY_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ==================================================
# SAVE PREDICTION
# ==================================================

def save_prediction(result, confidence):

    prediction = {
        "result": result.lower(),
        "confidence": round(confidence * 100, 2),
        "time": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    }

    history = get_prediction_history()

    history.append(prediction)

    with open(
        HISTORY_FILE,
        "w"
    ) as file:

        json.dump(
            history,
            file,
            indent=4
        )


# ==================================================
# GET HISTORY
# ==================================================

def get_prediction_history():

    if not HISTORY_FILE.exists():

        return []


    try:

        with open(
            HISTORY_FILE,
            "r"
        ) as file:

            return json.load(file)

    except (json.JSONDecodeError, FileNotFoundError):

        return []


# ==================================================
# CLEAR HISTORY
# ==================================================

def clear_prediction_history():

    if HISTORY_FILE.exists():

        HISTORY_FILE.unlink()