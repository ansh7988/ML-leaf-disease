import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("WEATHER_API_KEY")

def get_weather(location):
    url = "https://api.weatherapi.com/v1/current.json"

    params = {
        "key": API_KEY,
        "q": location,
        "aqi": "no"
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()

    data = response.json()

    return {
        "location": data["location"]["name"],
        "temperature": data["current"]["temp_c"],
        "humidity": data["current"]["humidity"],
        "precipitation": data["current"]["precip_mm"],
        "wind_speed": data["current"]["wind_kph"],
        "condition": data["current"]["condition"]["text"],
        "time": data["current"]["last_updated"]
    }