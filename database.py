# database.py

import sqlite3

DB_PATH = "detections.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT
        )
    ''')

    cursor.execute("PRAGMA table_info(images)")
    columns = [col[1] for col in cursor.fetchall()]
    
    if 'volume_liters' not in columns:
        cursor.execute("ALTER TABLE images ADD COLUMN volume_liters REAL")
        print("✅ 'volume_liters' column added.")
    else:
        print("ℹ️ 'volume_liters' column already exists.")

    if 'label' not in columns:
        cursor.execute("ALTER TABLE images ADD COLUMN label TEXT")
        print("✅ 'label' column added.")
    else:
        print("ℹ️ 'label' column already exists.")

    conn.commit()
    conn.close()


def insert_image_with_volume(image_path, volume_liters, label):
    image_path = image_path.replace("\\", "/")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO images (image_path, volume_liters, label)
        VALUES (?, ?, ?)
    """, (image_path, volume_liters, label))
    conn.commit()
    conn.close()
    print(f"[LOG] Saved to DB -> Image: {image_path}, Volume: {volume_liters} L, Label: {label}")


def fetch_all_images_with_volume_in_liters():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT image_path, volume_liters, label FROM images")
    rows = cursor.fetchall()
    conn.close()
    return [(img.replace("\\", "/"), vol, lbl) for img, vol, lbl in rows]


def query_images_by_param(param):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT image_path, volume_liters, label FROM images
        WHERE image_path LIKE ? OR label LIKE ? OR CAST(volume_liters AS TEXT) LIKE ?
    ''', (f'%{param}%', f'%{param}%', f'%{param}%'))
    results = cursor.fetchall()
    conn.close()
    return results
