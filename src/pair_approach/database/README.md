# SPARCS Database System

This directory contains the SQLite database implementation for the SPARCS pair approach, replacing CSV file storage with a more efficient database system.

## Overview

The database system consists of:
- **SQLite database** with tables for stars and pairs
- **Database management utilities** for inserting and querying data
- **Data loading scripts** to populate the database from the Hipparcos catalog
- **Updated catalog and IO modules** that use the database instead of CSV files

## Database Schema

### Stars Table
```sql
CREATE TABLE stars (
    hip INTEGER PRIMARY KEY,        -- HIP identifier
    raicrs REAL,                    -- Right ascension in ICRS
    deicrs REAL,                    -- Declination in ICRS
    vmag REAL,                      -- Visual magnitude
    ra_deg REAL,                    -- Right ascension in degrees
    dec_deg REAL,                   -- Declination in degrees
    x REAL,                         -- Unit vector X component
    y REAL,                         -- Unit vector Y component
    z REAL                          -- Unit vector Z component
);
```

### Pairs Table
```sql
CREATE TABLE pairs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    star1_id INTEGER,               -- First star HIP ID
    star2_id INTEGER,               -- Second star HIP ID
    angle REAL,                     -- Angular separation in degrees
    FOREIGN KEY (star1_id) REFERENCES stars(hip),
    FOREIGN KEY (star2_id) REFERENCES stars(hip)
);
```

## Quick Start

### 1. Setup Database and Load Data

Run the setup script to create the database and load the Hipparcos catalog:

```bash
cd src/pair_approach
python setup_and_load_data.py
```

This script will:
- Create the SQLite database with proper schema
- Load all stars from `HipparcosCatalog.txt` into the database
- Verify the data was loaded correctly
- Display statistics about the loaded data

### 2. Use the Database in Your Code

```python
from pair_approach.database import DatabaseManager

# Initialize database manager
db = DatabaseManager('database/star_catalog.db')

# Get stars by magnitude range
bright_stars = db.get_stars_by_magnitude_range(0.0, 3.0)

# Get stars in a field of view
fov_stars = db.get_stars_in_fov(center_ra=180.0, center_dec=0.0, fov_radius=15.0)

# Get a specific star
star = db.get_star_by_hip(32349)

# Get database statistics
star_count = db.get_star_count()
pair_count = db.get_pair_count()
```

## Files and Modules

### Core Database Files

- **`setup_db.py`**: Database creation and schema management
- **`db_utils.py`**: Database manager class and utility functions
- **`load_hipparcos_data.py`**: Script to load Hipparcos catalog into database
- **`__init__.py`**: Package initialization and exports

### Updated Modules

- **`catalog/catalog_loader.py`**: Updated to load from database instead of CSV
- **`utils/io.py`**: Updated to support database operations

### Setup Scripts

- **`setup_and_load_data.py`**: Complete setup and data loading script

## Database Operations

### Inserting Data

```python
from pair_approach.database import DatabaseManager

db = DatabaseManager('database/star_catalog.db')

# Insert a single star
db.insert_star(hip=32349, raicrs=101.29, deicrs=-16.71, vmag=-1.44,
               ra_deg=101.29, dec_deg=-16.71, x=0.123, y=0.456, z=0.789)

# Insert multiple stars in batch
stars_data = [
    (32349, 101.29, -16.71, -1.44, 101.29, -16.71, 0.123, 0.456, 0.789),
    (30438, 95.99, -52.70, -0.62, 95.99, -52.70, 0.234, 0.567, 0.890),
    # ... more stars
]
db.insert_stars_batch(stars_data)

# Insert star pairs
db.insert_pair(star1_id=32349, star2_id=30438, angle=15.5)
```

### Querying Data

```python
# Get stars by magnitude range
bright_stars = db.get_stars_by_magnitude_range(0.0, 3.0)

# Get stars in field of view
fov_stars = db.get_stars_in_fov(center_ra=180.0, center_dec=0.0, fov_radius=15.0)

# Get pairs by angle range
pairs = db.get_pairs_by_angle_range(10.0, 20.0)

# Get specific star
star = db.get_star_by_hip(32349)

# Get all stars as DataFrame
all_stars = get_stars_dataframe('database/star_catalog.db')
```

### Database Management

```python
# Get statistics
star_count = db.get_star_count()
pair_count = db.get_pair_count()

# Clear all data
db.clear_all_data()

# Reset database (drop and recreate tables)
from pair_approach.database import reset_database
reset_database('database/star_catalog.db')
```

## Migration from CSV

The updated modules maintain backward compatibility:

- **`catalog_loader.py`**: Tries database first, falls back to CSV if needed
- **`io.py`**: Keeps CSV functions for compatibility, adds database functions

### Before (CSV-based)
```python
from pair_approach.catalog.catalog_loader import load_hipparcos_catalog
hip = load_hipparcos_catalog('HipparcosCatalog.txt')
```

### After (Database-based)
```python
from pair_approach.catalog.catalog_loader import load_hipparcos_catalog
hip = load_hipparcos_catalog()  # Automatically uses database
```

## Performance Benefits

- **Faster queries**: Indexed database queries vs. CSV file parsing
- **Memory efficient**: Load only needed data instead of entire CSV
- **Structured data**: Proper data types and constraints
- **Concurrent access**: Multiple processes can access the database
- **Data integrity**: Foreign key constraints and data validation

## Troubleshooting

### Database Not Found
If you get "database not found" errors:
1. Run `python setup_and_load_data.py` to create the database
2. Check that the database file exists at `database/star_catalog.db`

### Import Errors
If you get import errors:
1. Make sure you're running from the `src/pair_approach` directory
2. Check that all `__init__.py` files exist
3. Verify Python path includes the `src` directory

### Data Loading Issues
If data loading fails:
1. Check that `HipparcosCatalog.txt` exists in the `src` directory
2. Verify the catalog file format matches the expected format
3. Check for sufficient disk space for the database file

## Next Steps

After completing this first stage (loading stars), the next steps will be:
1. Implement pair generation and storage
2. Update the matching algorithms to use database queries
3. Add more sophisticated querying capabilities
4. Implement caching for frequently accessed data 