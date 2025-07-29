import sqlite3
import os
from pathlib import Path

def create_database(db_path='star_catalog.db'):
    """
    Create the SQLite database with the required schema.
    
    Args:
        db_path (str): Path to the database file
    """
    # Ensure the database directory exists
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create stars table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stars (
            hip INTEGER PRIMARY KEY,
            raicrs REAL,
            deicrs REAL,
            vmag REAL,
            ra_deg REAL,
            dec_deg REAL,
            x REAL,
            y REAL,
            z REAL
        )
    ''')
    
    # Create pairs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS pairs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            star1_id INTEGER,
            star2_id INTEGER,
            angle REAL,
            FOREIGN KEY (star1_id) REFERENCES stars(hip),
            FOREIGN KEY (star2_id) REFERENCES stars(hip)
        )
    ''')
    
    # Create indexes for better performance
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_stars_vmag ON stars(vmag)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_pairs_star1 ON pairs(star1_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_pairs_star2 ON pairs(star2_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_pairs_angle ON pairs(angle)')
    
    conn.commit()
    conn.close()
    print(f"Database created successfully at {db_path}")

def reset_database(db_path='star_catalog.db'):
    """
    Reset the database by dropping all tables and recreating them.
    
    Args:
        db_path (str): Path to the database file
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Drop existing tables
    cursor.execute('DROP TABLE IF EXISTS pairs')
    cursor.execute('DROP TABLE IF EXISTS stars')
    
    conn.commit()
    conn.close()
    
    # Recreate the database
    create_database(db_path)
    print(f"Database reset successfully at {db_path}")

if __name__ == "__main__":
    # Create database in the pair_approach directory
    db_path = os.path.join(os.path.dirname(__file__), 'star_catalog.db')
    create_database(db_path) 