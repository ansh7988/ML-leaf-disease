def analyze_weather(weather):
    temperature = weather["temperature"]
    humidity = weather["humidity"]
    precipitation = weather["precipitation"]
    wind = weather["wind_speed"]
    condition = weather["condition"].lower()

    risks = []
    recommendations = []
    score = 0

    # -------------------------
    # HUMIDITY
    # -------------------------
    if humidity >= 90:
        score += 3
        risks.append("Very high humidity")
        recommendations.append(
            "Avoid unnecessary watering and improve air circulation."
        )

    elif humidity >= 75:
        score += 2
        risks.append("High humidity")
        recommendations.append(
            "Avoid unnecessary watering and keep plant leaves dry."
        )

    # -------------------------
    # PRECIPITATION
    # -------------------------
    if precipitation >= 10:
        score += 3
        risks.append("Heavy precipitation")
        recommendations.append(
            "Avoid additional watering and ensure proper drainage."
        )

    elif precipitation > 0:
        score += 1
        risks.append("Precipitation detected")
        recommendations.append(
            "Avoid unnecessary irrigation while the soil remains wet."
        )

    # -------------------------
    # WEATHER CONDITIONS
    # -------------------------
    moisture_conditions = [
        "rain",
        "drizzle",
        "mist",
        "fog"
    ]

    if any(condition_name in condition for condition_name in moisture_conditions):
        score += 1
        risks.append("Moist atmospheric conditions")
        recommendations.append(
            "Monitor leaves for prolonged moisture and improve air circulation."
        )

    # -------------------------
    # HIGH TEMPERATURE
    # -------------------------
    if temperature >= 35:
        score += 2
        risks.append("High temperature")
        recommendations.append(
            "Protect plants from excessive heat and monitor soil moisture."
        )

    # -------------------------
    # LOW TEMPERATURE
    # -------------------------
    elif temperature <= 10:
        score += 2
        risks.append("Low temperature")
        recommendations.append(
            "Protect sensitive plants from cold conditions."
        )

    # -------------------------
    # STRONG WIND
    # -------------------------
    if wind >= 30:
        score += 1
        risks.append("Strong wind")
        recommendations.append(
            "Protect plants from strong wind and secure weak stems."
        )

    # -------------------------
    # OVERALL RISK
    # -------------------------
    if score >= 6:
        risk_level = "HIGH"

    elif score >= 3:
        risk_level = "MODERATE"

    else:
        risk_level = "LOW"

    # -------------------------
    # GENERAL LOW-RISK MESSAGE
    # -------------------------
    if not recommendations:
        recommendations.append(
            "Weather conditions are currently relatively favorable. "
            "Continue normal plant care and monitor the plant regularly."
        )

    return {
        "risk_level": risk_level,
        "score": score,
        "risks": risks,
        "recommendations": recommendations
    }