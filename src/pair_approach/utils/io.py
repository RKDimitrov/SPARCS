import pandas as pd
import numpy as np
import os
from ..database import DatabaseManager

def save_vectors_to_csv(vectors, filename):
    """
    Save vectors to CSV file (kept for backward compatibility).
    
    Args:
        vectors (numpy.ndarray): Array of vectors
        filename (str): Output CSV filename
    """
    df = pd.DataFrame(vectors, columns=['x', 'y', 'z'])
    df.to_csv(filename, index=False)
    print(f"Vectors saved to CSV: {filename}")

def load_vectors_from_csv(filename):
    """
    Load vectors from CSV file (kept for backward compatibility).
    
    Args:
        filename (str): Input CSV filename
        
    Returns:
        numpy.ndarray: Array of vectors
    """
    df = pd.read_csv(filename)
    return df[['x', 'y', 'z']].values 

def save_vectors_to_database(vectors, db_path=None):
    """
    Save vectors to database as temporary detected stars.
    
    Args:
        vectors (numpy.ndarray): Array of vectors
        db_path (str, optional): Path to database file
    """
    if db_path is None:
        db_path = os.path.join(os.path.dirname(__file__), '../database/star_catalog.db')
    
    # For now, we'll store detected vectors in a temporary table
    # In a full implementation, you might want a separate table for detected stars
    print(f"Note: Detected vectors would be stored in database at {db_path}")
    print(f"Number of vectors: {len(vectors)}")

def load_vectors_from_database(db_path=None):
    """
    Load vectors from database.
    
    Args:
        db_path (str, optional): Path to database file
        
    Returns:
        numpy.ndarray: Array of vectors
    """
    if db_path is None:
        db_path = os.path.join(os.path.dirname(__file__), '../database/star_catalog.db')
    
    db = DatabaseManager(db_path)
    conn = db.get_connection()
    df = pd.read_sql_query('SELECT x, y, z FROM stars', conn)
    conn.close()
    
    return df[['x', 'y', 'z']].values

def save_matches_to_database(matches, db_path=None):
    """
    Save star matches to database.
    
    Args:
        matches (list): List of match dictionaries
        db_path (str, optional): Path to database file
    """
    if db_path is None:
        db_path = os.path.join(os.path.dirname(__file__), '../database/star_catalog.db')
    
    print(f"Note: Matches would be stored in database at {db_path}")
    print(f"Number of matches: {len(matches)}")

def load_matches_from_database(db_path=None):
    """
    Load star matches from database.
    
    Args:
        db_path (str, optional): Path to database file
        
    Returns:
        list: List of match dictionaries
    """
    if db_path is None:
        db_path = os.path.join(os.path.dirname(__file__), '../database/star_catalog.db')
    
    # This would load matches from a matches table
    # For now, return empty list
    return [] 