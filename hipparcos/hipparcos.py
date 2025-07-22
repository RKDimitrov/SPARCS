input_file = 'HipparcosCatalog.txt'
output_file = 'hipparcos_stars_bright.txt'

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    outfile.write('|name      |ra           |dec          |pm_ra   |pm_dec  |parallax|spect_type  |vmag |\n')

    for line in infile:
        if not line.strip() or not line.startswith('H|'):
            continue

        parts = [p.strip() for p in line.split('|')]

        try:
            hip = parts[1]
            ra = parts[3]
            dec = parts[4]
            vmag = float(parts[5])
            pm_ra = float(parts[10])
            pm_dec = float(parts[11])
            parallax = float(parts[9])
            spect_type = parts[-2]  # Spectral type is usually near the end

            if vmag < 5:
                name = f'HIP {hip}'
                outfile.write(f'|{name:<10}|{ra:<13}|{dec:<13}|{pm_ra:8.2f}|{pm_dec:8.2f}|{parallax:8.2f}|{spect_type:<12}|{vmag:5.2f}|\n')

        except (IndexError, ValueError):
            continue  # Skip malformed lines

print(f"Extraction complete. Results written to {output_file}.")
