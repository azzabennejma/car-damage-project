import sqlite3
from passlib.context import CryptContext

pwd = CryptContext(schemes=["bcrypt"])

conn = sqlite3.connect("users.db")
cursor = conn.cursor()

cursor.execute("""
INSERT OR IGNORE INTO users (username, email, password, role)
VALUES (?, ?, ?, ?)
""", (
    "admin",
    "admin@gmail.com",
    pwd.hash("admin123"),
    "admin"
))

conn.commit()
conn.close()

print("Admin created")