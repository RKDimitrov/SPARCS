input_file = 'hipparcosData1.txt'
output_file = 'hipparcos_stars_bright.txt'

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    outfile.write('HIP\tRA\tDec\tVmag\n')
    for line in infile:
        if not line.strip() or not line.startswith('H|'):
            continue  
        parts = [p.strip() for p in line.split('|')]
        try:
            hip = parts[1]
            ra = parts[3]
            dec = parts[4]
            vmag = float(parts[5])
            if vmag < 5:
                outfile.write(f'{hip}\t{ra}\t{dec}\t{vmag}\n')
        except (IndexError, ValueError):
            continue  

print(f"Extraction complete. Results written to {output_file}.")
