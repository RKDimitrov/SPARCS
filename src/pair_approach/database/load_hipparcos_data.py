import pandas as pd
import numpy as np
import os
import re
from typing import List, Tuple
from .setup_db import create_database
from .db_utils import DatabaseManager

def parse_hipparcos_line(line: str) -> Tuple[int, float, float, float]:
    """
    Parse a line from the Hipparcos catalog.
    
    Args:
        line (str): Line from the catalog file
        
    Returns:
        Tuple of (hip, ra_deg, dec_deg, vmag)
    """
    # Remove leading/trailing whitespace and split by |
    parts = [part.strip() for part in line.split('|')]
    
    if len(parts) < 9:  # Need at least 9 parts (including empty first part)
        return None
    
    try:
        # Extract HIP number (remove 'HIP ' prefix)
        hip_str = parts[1]  # First data column is at index 1
        if not hip_str or hip_str == 'name':  # Skip header or empty lines
            return None
        if hip_str.startswith('HIP '):
            hip = int(hip_str[4:])
        else:
            hip = int(hip_str)
        
        # Parse right ascension (format: "06 45 09.2499")
        ra_str = parts[2]
        if not ra_str:
            return None
        ra_deg = hms_to_deg(ra_str)
        
        # Parse declination (format: "-16 42 47.315")
        dec_str = parts[3]
        if not dec_str:
            return None
        dec_deg = dms_to_deg(dec_str)
        
        # Parse visual magnitude
        vmag_str = parts[8]
        if not vmag_str:
            return None
        vmag = float(vmag_str)
        
        return hip, ra_deg, dec_deg, vmag
        
    except (ValueError, IndexError) as e:
        print(f"Error parsing line: {line.strip()}")
        print(f"Error: {e}")
        return None

def hms_to_deg(hms_str: str) -> float:
    """
    Convert hours:minutes:seconds to degrees.
    
    Args:
        hms_str (str): String in format "HH MM SS.SSSS"
        
    Returns:
        float: Right ascension in degrees
    """
    parts = hms_str.split()
    h = float(parts[0])
    m = float(parts[1])
    s = float(parts[2])
    return 15 * (h + m/60 + s/3600)  # Convert to degrees (15 degrees per hour)

def dms_to_deg(dms_str: str) -> float:
    """
    Convert degrees:minutes:seconds to degrees.
    
    Args:
        dms_str (str): String in format "DD MM SS.SSSS" or "-DD MM SS.SSSS"
        
    Returns:
        float: Declination in degrees
    """
    parts = dms_str.split()
    sign = -1 if parts[0].startswith('-') else 1
    deg = abs(float(parts[0]))
    m = float(parts[1])
    s = float(parts[2])
    return sign * (deg + m/60 + s/3600)

def calculate_unit_vectors(ra_deg: float, dec_deg: float) -> Tuple[float, float, float]:
    """
    Calculate unit vector components from right ascension and declination.
    
    Args:
        ra_deg (float): Right ascension in degrees
        dec_deg (float): Declination in degrees
        
    Returns:
        Tuple of (x, y, z) unit vector components
    """
    ra_rad = np.deg2rad(ra_deg)
    dec_rad = np.deg2rad(dec_deg)
    
    x = np.cos(dec_rad) * np.cos(ra_rad)
    y = np.cos(dec_rad) * np.sin(ra_rad)
    z = np.sin(dec_rad)
    
    return x, y, z

def load_hipparcos_to_database(catalog_path: str, db_path: str = 'star_catalog.db'):
    """
    Load Hipparcos catalog data into the SQLite database.
    
    Args:
        catalog_path (str): Path to the Hipparcos catalog file
        db_path (str): Path to the SQLite database file
    """
    # Ensure database exists
    create_database(db_path)
    
    # Initialize database manager
    db = DatabaseManager(db_path)
    
    # Clear existing data
    db.clear_all_data()
    
    # Read and parse the catalog file
    stars_data = []
    line_count = 0
    success_count = 0
    
    print(f"Loading Hipparcos catalog from {catalog_path}...")
    
    with open(catalog_path, 'r') as file:
        for line in file:
            line_count += 1
            
            # Parse the line (header will be skipped in parse_hipparcos_line)
            parsed = parse_hipparcos_line(line)
            if parsed is None:
                continue
            
            hip, ra_deg, dec_deg, vmag = parsed
            
            # Calculate unit vectors
            x, y, z = calculate_unit_vectors(ra_deg, dec_deg)
            
            # Prepare data for database insertion
            # Note: raicrs and deicrs are the same as ra_deg and dec_deg for this catalog
            star_data = (hip, ra_deg, dec_deg, vmag, ra_deg, dec_deg, x, y, z)
            stars_data.append(star_data)
            success_count += 1
            
            # Progress indicator
            if success_count % 100 == 0:
                print(f"Processed {success_count} stars...")
    
    # Insert all stars in batch
    print(f"Inserting {len(stars_data)} stars into database...")
    db.insert_stars_batch(stars_data)
    
    # Verify insertion
    final_count = db.get_star_count()
    print(f"Successfully loaded {final_count} stars into database")
    
    # Print some statistics
    if final_count > 0:
        # Get magnitude range
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT MIN(vmag), MAX(vmag), AVG(vmag) FROM stars')
        min_mag, max_mag, avg_mag = cursor.fetchone()
        conn.close()
        
        print(f"Magnitude range: {min_mag:.2f} to {max_mag:.2f}")
        print(f"Average magnitude: {avg_mag:.2f}")

def main():
    """Main function to load Hipparcos data."""
    # Get the path to the Hipparcos catalog
    catalog_path = os.path.join(os.path.dirname(__file__), '../HipparcosCatalog.txt')
    db_path = os.path.join(os.path.dirname(__file__), 'star_catalog.db')
    
    if not os.path.exists(catalog_path):
        print(f"Error: Catalog file not found at {catalog_path}")
        return
    
    load_hipparcos_to_database(catalog_path, db_path)

if __name__ == "__main__":
    main() 