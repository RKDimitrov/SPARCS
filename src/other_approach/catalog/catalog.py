import pandas as pd
import numpy as np

def hms_to_deg(hms_str):
    h, m, s = map(float, hms_str.strip().split())
    return 15 * (h + m / 60 + s / 3600)

def dms_to_deg(dms_str):
    sign = -1 if dms_str.strip().startswith('-') else 1
    parts = dms_str.strip().replace('+', '').replace('-', '').split()
    d, m, s = map(float, parts)
    return sign * (d + m / 60 + s / 3600)

def load_hipparcos_catalog(txt_path, mag_limit=7):
    df = pd.read_csv(txt_path, sep='|', engine='python')
    df.columns = [c.strip() for c in df.columns]
    df['ra_deg'] = df['ra'].apply(hms_to_deg)
    df['dec_deg'] = df['dec'].apply(dms_to_deg)
    df['vmag'] = pd.to_numeric(df['vmag'], errors='coerce')
    df = df[df['vmag'] <= mag_limit].copy()
    ra_rad = np.deg2rad(df['ra_deg'])
    dec_rad = np.deg2rad(df['dec_deg'])
    df['x'] = np.cos(dec_rad) * np.cos(ra_rad)
    df['y'] = np.cos(dec_rad) * np.sin(ra_rad)
    df['z'] = np.sin(dec_rad)
    return df 