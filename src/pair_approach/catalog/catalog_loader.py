import pandas as pd
import numpy as np

def load_hipparcos_catalog(catalog_path):
    """Load Hipparcos catalog and return DataFrame with relevant columns."""
    hip = pd.read_csv(catalog_path, sep='|')
    hip.columns = hip.columns.str.strip()
    return hip

def hms_to_deg(hms_str):
    h, m, s = [float(part) for part in hms_str.split()]
    return 15 * (h + m/60 + s/3600)

def dms_to_deg(dms_str):
    parts = dms_str.split()
    sign = -1 if parts[0].startswith('-') else 1
    deg = float(parts[0].replace('+','').replace('-',''))
    m = float(parts[1])
    s = float(parts[2])
    return sign * (deg + m/60 + s/3600)

def add_catalog_unit_vectors(hip):
    hip['ra_deg'] = hip['ra'].apply(hms_to_deg)
    hip['dec_deg'] = hip['dec'].apply(dms_to_deg)
    hip['ra_rad'] = np.deg2rad(hip['ra_deg'])
    hip['dec_rad'] = np.deg2rad(hip['dec_deg'])
    hip['x'] = np.cos(hip['dec_rad']) * np.cos(hip['ra_rad'])
    hip['y'] = np.cos(hip['dec_rad']) * np.sin(hip['ra_rad'])
    hip['z'] = np.sin(hip['dec_rad'])
    return hip 