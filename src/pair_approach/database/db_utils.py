import sqlite3
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import os

class DatabaseManager:
    """Database manager for star catalog operations."""
    
    def __init__(self, db_path='star_catalog.db'):
        """
        Initialize database manager.
        
        Args:
            db_path (str): Path to the SQLite database file
        """
        self.db_path = db_path
    
    def get_connection(self):
        """Get a database connection."""
        return sqlite3.connect(self.db_path)
    
    def insert_star(self, hip: int, raicrs: float, deicrs: float, vmag: float, 
                   ra_deg: float, dec_deg: float, x: float, y: float, z: float):
        """
        Insert a single star into the database.
        
        Args:
            hip (int): HIP identifier
            raicrs (float): Right ascension in ICRS
            deicrs (float): Declination in ICRS
            vmag (float): Visual magnitude
            ra_deg (float): Right ascension in degrees
            dec_deg (float): Declination in degrees
            x, y, z (float): Unit vector components
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO stars 
                (hip, raicrs, deicrs, vmag, ra_deg, dec_deg, x, y, z)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (hip, raicrs, deicrs, vmag, ra_deg, dec_deg, x, y, z))
            conn.commit()
        except Exception as e:
            print(f"Error inserting star {hip}: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def insert_stars_batch(self, stars_data: List[Tuple]):
        """
        Insert multiple stars in a batch operation.
        
        Args:
            stars_data: List of tuples containing star data
                       (hip, raicrs, deicrs, vmag, ra_deg, dec_deg, x, y, z)
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.executemany('''
                INSERT OR REPLACE INTO stars 
                (hip, raicrs, deicrs, vmag, ra_deg, dec_deg, x, y, z)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', stars_data)
            conn.commit()
            print(f"Inserted {len(stars_data)} stars successfully")
        except Exception as e:
            print(f"Error in batch insert: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def insert_pair(self, star1_id: int, star2_id: int, angle: float):
        """
        Insert a star pair into the database.
        
        Args:
            star1_id (int): First star HIP ID
            star2_id (int): Second star HIP ID
            angle (float): Angular separation in degrees
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO pairs (star1_id, star2_id, angle)
                VALUES (?, ?, ?)
            ''', (star1_id, star2_id, angle))
            conn.commit()
        except Exception as e:
            print(f"Error inserting pair {star1_id}-{star2_id}: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def insert_pairs_batch(self, pairs_data: List[Tuple]):
        """
        Insert multiple pairs in a batch operation.
        
        Args:
            pairs_data: List of tuples containing pair data (star1_id, star2_id, angle)
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.executemany('''
                INSERT INTO pairs (star1_id, star2_id, angle)
                VALUES (?, ?, ?)
            ''', pairs_data)
            conn.commit()
            print(f"Inserted {len(pairs_data)} pairs successfully")
        except Exception as e:
            print(f"Error in batch insert pairs: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def get_star_by_hip(self, hip: int) -> Optional[Dict[str, Any]]:
        """
        Get a star by its HIP ID.
        
        Args:
            hip (int): HIP identifier
            
        Returns:
            Dictionary with star data or None if not found
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM stars WHERE hip = ?', (hip,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            columns = ['hip', 'raicrs', 'deicrs', 'vmag', 'ra_deg', 'dec_deg', 'x', 'y', 'z']
            return dict(zip(columns, row))
        return None
    
    def get_stars_by_magnitude_range(self, min_mag: float, max_mag: float) -> pd.DataFrame:
        """
        Get stars within a magnitude range.
        
        Args:
            min_mag (float): Minimum magnitude
            max_mag (float): Maximum magnitude
            
        Returns:
            DataFrame with star data
        """
        conn = self.get_connection()
        query = '''
            SELECT hip, raicrs, deicrs, vmag, ra_deg, dec_deg, x, y, z
            FROM stars 
            WHERE vmag BETWEEN ? AND ?
            ORDER BY vmag
        '''
        df = pd.read_sql_query(query, conn, params=(min_mag, max_mag))
        conn.close()
        return df
    
    def get_stars_in_fov(self, center_ra: float, center_dec: float, fov_radius: float) -> pd.DataFrame:
        """
        Get stars within a field of view.
        
        Args:
            center_ra (float): Center right ascension in degrees
            center_dec (float): Center declination in degrees
            fov_radius (float): Field of view radius in degrees
            
        Returns:
            DataFrame with star data
        """
        conn = self.get_connection()
        query = '''
            SELECT hip, raicrs, deicrs, vmag, ra_deg, dec_deg, x, y, z
            FROM stars 
            WHERE ABS(ra_deg - ?) <= ? AND ABS(dec_deg - ?) <= ?
            ORDER BY vmag
        '''
        df = pd.read_sql_query(query, conn, params=(center_ra, fov_radius, center_dec, fov_radius))
        conn.close()
        return df
    
    def get_pairs_by_angle_range(self, min_angle: float, max_angle: float) -> pd.DataFrame:
        """
        Get pairs within an angle range.
        
        Args:
            min_angle (float): Minimum angle in degrees
            max_angle (float): Maximum angle in degrees
            
        Returns:
            DataFrame with pair data
        """
        conn = self.get_connection()
        query = '''
            SELECT p.id, p.star1_id, p.star2_id, p.angle,
                   s1.vmag as star1_vmag, s2.vmag as star2_vmag
            FROM pairs p
            JOIN stars s1 ON p.star1_id = s1.hip
            JOIN stars s2 ON p.star2_id = s2.hip
            WHERE p.angle BETWEEN ? AND ?
            ORDER BY p.angle
        '''
        df = pd.read_sql_query(query, conn, params=(min_angle, max_angle))
        conn.close()
        return df
    
    def get_star_count(self) -> int:
        """Get total number of stars in the database."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM stars')
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def get_pair_count(self) -> int:
        """Get total number of pairs in the database."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM pairs')
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def clear_all_data(self):
        """Clear all data from the database."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('DELETE FROM pairs')
            cursor.execute('DELETE FROM stars')
            conn.commit()
            print("All data cleared from database")
        except Exception as e:
            print(f"Error clearing data: {e}")
            conn.rollback()
        finally:
            conn.close()

# Convenience functions for backward compatibility
def get_db_manager(db_path='star_catalog.db') -> DatabaseManager:
    """Get a database manager instance."""
    return DatabaseManager(db_path)

def insert_star_data(hip: int, raicrs: float, deicrs: float, vmag: float, 
                    ra_deg: float, dec_deg: float, x: float, y: float, z: float,
                    db_path='star_catalog.db'):
    """Convenience function to insert a single star."""
    db = DatabaseManager(db_path)
    db.insert_star(hip, raicrs, deicrs, vmag, ra_deg, dec_deg, x, y, z)

def get_stars_dataframe(db_path='star_catalog.db') -> pd.DataFrame:
    """Get all stars as a DataFrame."""
    db = DatabaseManager(db_path)
    conn = db.get_connection()
    df = pd.read_sql_query('SELECT * FROM stars', conn)
    conn.close()
    return df 