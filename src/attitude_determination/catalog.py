import pandas as pd

class HipparcosCatalog:
    def __init__(self, catalog_file):
        self.catalog = self._load_catalog(catalog_file)

    def _load_catalog(self, catalog_file):
        rows = []
        try:
            with open(catalog_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if 'HIP' not in line or '|' not in line:
                        continue
                    parts = [p.strip() for p in line.split('|') if p.strip()]
                    try:
                        hip_id = int(parts[0].split()[1])
                        ra_h, ra_m, ra_s = [float(x) for x in parts[1].split()]
                        ra_deg = (ra_h + ra_m/60.0 + ra_s/3600.0) * 15.0
                        dec_parts = parts[2].split()
                        sign = -1 if dec_parts[0].startswith('-') else 1
                        dec_d = int(dec_parts[0].lstrip('+-'))
                        dec_m = int(dec_parts[1]); dec_s = float(dec_parts[2])
                        dec_deg = sign * (abs(dec_d) + dec_m/60.0 + dec_s/3600.0)
                        rows.append({'HIP': hip_id, 'RA_deg': ra_deg, 'Dec_deg': dec_deg})
                    except Exception:
                        continue
            return pd.DataFrame(rows)
        except Exception as e:
            print(f"Error loading catalog: {e}")
            return pd.DataFrame()

    def get_star_coords(self, hip_id):
        star = self.catalog[self.catalog["HIP"] == hip_id]
        return (None, None) if len(star) == 0 else (star.iloc[0]["RA_deg"], star.iloc[0]["Dec_deg"])
