import sqlite3

def init_db():
    conn = sqlite3.connect("detections.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT,
            confidence REAL,
            image_path TEXT
        )
    ''')
    conn.commit()
    conn.close()

def insert_detection(label, confidence, image_path):
    conn = sqlite3.connect("detections.db")
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO detections (label, confidence, image_path)
        VALUES (?, ?, ?)
    ''', (label, confidence, image_path))
    conn.commit()
    conn.close()

    print(f"✅ Inserted into DB -> Label: {label}, Confidence: {confidence:.2f}, Image: {image_path}")

