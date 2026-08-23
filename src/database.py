import os
import streamlit as st
import psycopg
from dotenv import load_dotenv

load_dotenv()

def get_connection():
    database_url = os.getenv("DATABASE_URL")
    if not database_url: database_url = st.secrets["DATABASE_URL"]

    return psycopg.connect(database_url)

def get_users():
    conn = get_connection()

    cursor = conn.cursor()
    cursor.execute("SELECT id, name, email FROM users")

    users = cursor.fetchall()

    cursor.close()
    conn.close()

    return users


if __name__ == "__main__":
    print(get_users())

import bcrypt


def hash_password(password):
    return bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt()
    ).decode("utf-8")

if __name__ == "__main__":
    hashed = hash_password("mypassword123")
    print(hashed)

def verify_password(password, stored_hash):
    return bcrypt.checkpw(
        password.encode("utf-8"),
        stored_hash.encode("utf-8")
    )

def create_user(name, email, password):

    conn = get_connection()
    cur = conn.cursor()

    hashed_password = hash_password(password)

    cur.execute(
        """
        INSERT INTO users (name, email, password_hash)
        VALUES (%s, %s, %s)
        """,
        (name, email, hashed_password)
    )

    conn.commit()

    cur.close()
    conn.close()

    print("User created successfully!")

def login_user(email, password):

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT id, name, password_hash
        FROM users
        WHERE email = %s
        """,
        (email,)
    )

    user = cur.fetchone()

    cur.close()
    conn.close()

    if user is None:
        return None

    user_id, name, stored_hash = user

    if verify_password(password, stored_hash):
        return {
            "id": user_id,
            "name": name
        }

    return None

def save_prediction(user_id, image_path, predicted_disease, confidence):

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO predictions
        (user_id, image_path, predicted_disease, confidence)
        VALUES (%s, %s, %s, %s)
        """,
        (
            user_id,
            image_path,
            predicted_disease,
            confidence
        )
    )

    conn.commit()

    cur.close()
    conn.close()

def get_user_predictions(user_id):

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT predicted_disease, confidence, created_at
        FROM predictions
        WHERE user_id = %s
        ORDER BY created_at DESC
        """,
        (user_id,)
    )

    rows = cur.fetchall()

    cur.close()
    conn.close()

    history = []

    for disease, confidence, created_at in rows:
        history.append({
            "result": disease,
            "confidence": float(confidence) * 100,
            "time": created_at.strftime("%Y-%m-%d %H:%M:%S")
        })

    return history

def clear_user_predictions(user_id):

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        DELETE FROM predictions
        WHERE user_id = %s
        """,
        (user_id,)
    )

    conn.commit()

    cur.close()
    conn.close()