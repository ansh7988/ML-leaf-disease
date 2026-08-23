import os
import streamlit as st
import psycopg
import re
import psycopg.errors

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

    try:
        hashed_password = hash_password(password)

        cur.execute(
            """
            INSERT INTO users (name, email, password_hash)
            VALUES (%s, %s, %s)
            RETURNING id, name
            """,
            (name, email, hashed_password)
        )

        new_user = cur.fetchone()

        conn.commit()

        return True, {
            "id": new_user[0],
            "name": new_user[1]
        }

    except psycopg.errors.UniqueViolation:
        conn.rollback()

        return False, (
            "An account with this email already exists. Please sign in instead."
        )

    except Exception:
        conn.rollback()

        return False, "Unable to create account. Please try again."

    finally:
        cur.close()
        conn.close()

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

# ==================================================
# USER PROFILE FUNCTIONS
# ==================================================

def get_user_profile(user_id):
    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute(
            """
            SELECT id, name, email
            FROM users
            WHERE id = %s
            """,
            (user_id,)
        )

        user = cur.fetchone()

        if user is None:
            return None

        return {
            "id": user[0],
            "name": user[1],
            "email": user[2]
        }

    finally:
        cur.close()
        conn.close()


def update_user_name(user_id, new_name):
    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute(
            """
            UPDATE users
            SET name = %s
            WHERE id = %s
            """,
            (new_name, user_id)
        )

        if cur.rowcount == 0:
            conn.rollback()
            return False, "User account was not found."

        conn.commit()

        return True, "Your name has been updated successfully."

    except Exception:
        conn.rollback()
        return False, "Unable to update your name. Please try again."

    finally:
        cur.close()
        conn.close()


def update_user_password(user_id, current_password, new_password):
    conn = get_connection()
    cur = conn.cursor()

    try:
        # Get the current password hash
        cur.execute(
            """
            SELECT password_hash
            FROM users
            WHERE id = %s
            """,
            (user_id,)
        )

        user = cur.fetchone()

        if user is None:
            return False, "User account was not found."

        stored_hash = user[0]

        # Verify current password first
        if not verify_password(current_password, stored_hash):
            return False, "Your current password is incorrect."

        # Hash new password
        new_hashed_password = hash_password(new_password)

        cur.execute(
            """
            UPDATE users
            SET password_hash = %s
            WHERE id = %s
            """,
            (new_hashed_password, user_id)
        )

        conn.commit()

        return True, "Your password has been updated successfully."

    except Exception:
        conn.rollback()
        return False, "Unable to update your password. Please try again."

    finally:
        cur.close()
        conn.close()