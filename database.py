import sqlite3

def init_db():
    conn = sqlite3.connect("detections.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def insert_image_path(image_path):
    conn = sqlite3.connect("detections.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT
        )
    ''')
    cursor.execute('INSERT INTO images (image_path) VALUES (?)', (image_path,))
    conn.commit()
    conn.close()

    print(f"✅ Image path stored in DB: {image_path}")
