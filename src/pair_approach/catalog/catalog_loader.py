import pandas as pd
import numpy as np
import os
from ..database import DatabaseManager

def load_hipparcos_catalog(catalog_path=None, db_path=None):
    """
    Load Hipparcos catalog from database or CSV file.
    
    Args:
        catalog_path (str, optional): Path to CSV catalog file (for backward compatibility)
        db_path (str, optional): Path to SQLite database file
        
    Returns:
        pandas.DataFrame: Catalog data with columns including ra_deg, dec_deg, x, y, z
    """
    if db_path is None:
        # Default database path in the database directory
        db_path = os.path.join(os.path.dirname(__file__), '../database/star_catalog.db')
    
    # Try to load from database first
    if os.path.exists(db_path):
        try:
            db = DatabaseManager(db_path)
            conn = db.get_connection()
            hip = pd.read_sql_query('SELECT * FROM stars', conn)
            conn.close()
            
            if not hip.empty:
                print(f"Loaded {len(hip)} stars from database")
                return hip
        except Exception as e:
            print(f"Error loading from database: {e}")
    
    # Fallback to CSV loading (for backward compatibility)
    if catalog_path and os.path.exists(catalog_path):
        print("Loading from CSV file (database not available)")
    hip = pd.read_csv(catalog_path, sep='|')
    hip.columns = hip.columns.str.strip()
    return hip
    else:
        raise FileNotFoundError("Neither database nor CSV file found")

def hms_to_deg(hms_str):
    """Convert hours:minutes:seconds to degrees."""
    h, m, s = [float(part) for part in hms_str.split()]
    return 15 * (h + m/60 + s/3600)

def dms_to_deg(dms_str):
    """Convert degrees:minutes:seconds to degrees."""
    parts = dms_str.split()
    sign = -1 if parts[0].startswith('-') else 1
    deg = float(parts[0].replace('+','').replace('-',''))
    m = float(parts[1])
    s = float(parts[2])
    return sign * (deg + m/60 + s/3600)

def add_catalog_unit_vectors(hip):
    """
    Add unit vector columns to catalog DataFrame.
    
    Args:
        hip (pandas.DataFrame): Catalog DataFrame
        
    Returns:
        pandas.DataFrame: Catalog DataFrame with unit vector columns
    """
    # Check if unit vectors already exist
    if 'x' in hip.columns and 'y' in hip.columns and 'z' in hip.columns:
        print("Unit vectors already present in catalog")
        return hip
    
    # Check if we have ra_deg and dec_deg
    if 'ra_deg' not in hip.columns or 'dec_deg' not in hip.columns:
        # Convert from string format if needed
        if 'ra' in hip.columns and 'dec' in hip.columns:
    hip['ra_deg'] = hip['ra'].apply(hms_to_deg)
    hip['dec_deg'] = hip['dec'].apply(dms_to_deg)
        else:
            raise ValueError("No right ascension or declination data found")
    
    # Calculate unit vectors
    hip['ra_rad'] = np.deg2rad(hip['ra_deg'])
    hip['dec_rad'] = np.deg2rad(hip['dec_deg'])
    hip['x'] = np.cos(hip['dec_rad']) * np.cos(hip['ra_rad'])
    hip['y'] = np.cos(hip['dec_rad']) * np.sin(hip['ra_rad'])
    hip['z'] = np.sin(hip['dec_rad'])
    
    return hip 

def get_catalog_subset(hip, min_mag=None, max_mag=None, fov_center=None, fov_radius=None):
    """
    Get a subset of the catalog based on magnitude and field of view constraints.
    
    Args:
        hip (pandas.DataFrame): Full catalog DataFrame
        min_mag (float, optional): Minimum magnitude
        max_mag (float, optional): Maximum magnitude
        fov_center (tuple, optional): (ra_deg, dec_deg) center of field of view
        fov_radius (float, optional): Field of view radius in degrees
        
    Returns:
        pandas.DataFrame: Filtered catalog subset
    """
    subset = hip.copy()
    
    # Filter by magnitude
    if min_mag is not None:
        subset = subset[subset['vmag'] >= min_mag]
    if max_mag is not None:
        subset = subset[subset['vmag'] <= max_mag]
    
    # Filter by field of view
    if fov_center is not None and fov_radius is not None:
        center_ra, center_dec = fov_center
        ra_diff = np.abs(subset['ra_deg'] - center_ra)
        dec_diff = np.abs(subset['dec_deg'] - center_dec)
        subset = subset[(ra_diff <= fov_radius) & (dec_diff <= fov_radius)]
    
    return subset 