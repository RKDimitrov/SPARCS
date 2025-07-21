import requests
import json

STELLARIUM_API_URL = 'http://localhost:8090/api/objects/info'
OUTPUT_FILE = 'stars_vmag_lt5_locations.txt'

BRIGHT_STARS = [
    'Sirius', 'Canopus', 'Arcturus', 'Vega', 'Capella', 'Rigel', 'Procyon', 'Achernar', 'Betelgeuse', 'Hadar',
    'Altair', 'Aldebaran', 'Antares', 'Spica', 'Pollux', 'Fomalhaut', 'Deneb', 'Mimosa', 'Regulus', 'Adhara'
]

def get_star_info(star_name):
    params = {'name': star_name, 'format': 'json'}
    try:
        response = requests.get(STELLARIUM_API_URL, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": f"Error fetching info for {star_name}: {e}"}

def main():
    results = []
    for star in BRIGHT_STARS:
        info = get_star_info(star)
        results.append({"star": star, "info": info})
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    main()
