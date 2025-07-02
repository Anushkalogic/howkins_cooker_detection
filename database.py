# database.py

import sqlite3

DB_PATH = "detections.db"
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT,
            volume_liters REAL,
            label TEXT,
            unique_id TEXT
        )
    ''')

    # Check & add if column not already added (optional safety)
    cursor.execute("PRAGMA table_info(images)")
    columns = [col[1] for col in cursor.fetchall()]
    
    if 'unique_id' not in columns:
        cursor.execute("ALTER TABLE images ADD COLUMN unique_id TEXT")
        print("✅ 'unique_id' column added.")

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

    last_id = cursor.lastrowid
    unique_id = f"hw{last_id}"

    # ✅ Now update the row with the generated ID
    cursor.execute("""
        UPDATE images SET unique_id = ? WHERE id = ?
    """, (unique_id, last_id))
    conn.commit()

    print("📥 INSERTED TO DB:")
    print(f"    🆔 Unique ID   : {unique_id}")
    print(f"    📷 Image Path : {image_path}")
    print(f"    📦 Volume (L) : {volume_liters}")
    print(f"    🏷️ Label      : {label}")

    conn.close()

    return unique_id  # ✅ return this value

# database.py
def fetch_all_images_with_volume_in_liters():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT image_path, volume_liters, label, unique_id FROM images")
    rows = cursor.fetchall()
    conn.close()
    return [(img.replace("\\", "/"), vol, lbl, uid) for img, vol, lbl, uid in rows]


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

def update_defect_entries():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Set label = 'defect' where it's NULL or empty
    cursor.execute("""
        UPDATE images
        SET label = 'defect'
        WHERE label IS NULL OR TRIM(label) = ''
    """)

    # Set volume_liters = 0.0 where it's NULL
    cursor.execute("""
        UPDATE images
        SET volume_liters = 0.0
        WHERE volume_liters IS NULL
    """)

    conn.commit()
    print("🛠️ Updated: Set NULL labels to 'defect' and NULL volumes to 0.0")
    conn.close()

def cleanup_null_entries():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM images WHERE volume_liters IS NULL OR label IS NULL")
    conn.commit()
    conn.close()
    print("🧹 Deleted entries with NULL volume or label.")
