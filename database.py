import sqlite3

DB_NAME = "detections.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Create table with proper columns
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT,
            volume REAL,
            label TEXT
        )
    ''')

    conn.commit()
    conn.close()


def insert_image_with_volume(image_path, volume, label=None):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO images (image_path, volume, label)
        VALUES (?, ?, ?)
    ''', (image_path, volume, label))
    conn.commit()
    conn.close()
    print(f"✅ DB Entry: {image_path}, Volume: {volume}, Label: {label}")


def fetch_all_images_with_volume():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT image_path, volume FROM images")
    results = cursor.fetchall()
    conn.close()
    return results


def query_images_by_param(param):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT image_path, volume FROM images
        WHERE image_path LIKE ?
        OR CAST(volume AS TEXT) LIKE ?
        OR label LIKE ?
    ''', (f'%{param}%', f'%{param}%', f'%{param}%'))
    results = cursor.fetchall()
    conn.close()
    return results
