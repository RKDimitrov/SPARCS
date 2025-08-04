# S.P.A.R.C.S  
**Space Pointing Attitude Recognition Control System**

## Team Members
- Kiiran Jadhav (Team Leader)  
- Mubasshirah Khan
- Fabrizio Marin
- Dario Maione
- Radoslav Dimitrov

## Goal  
A compact star tracker system that uses computer vision to recognize stars and determine spacecraft attitude.

## Current Design Decisions
- **Hardware**: Raspberry Pi 4 with Raspberry Pi AI camera and BNO055 IMU sensor  
- **Image Processing**: AI camera captures star field images and processes them locally on the Pi  
- **Sensor Fusion**: Star data is combined with IMU readings to improve orientation accuracy  
- **Modular Architecture**: Separate modules for image capture, star detection from the image, combination between star pairs from the star catalog, combination between stars from the image, matching between the two pairs and finally using Quest method to calculate the attitude.

## Open Questions
- How accurate is the attitude estimation under different lighting conditions?
- Can processing speed and latency be further optimized for real-time applications?

---

## How to Run the Modular Pair Approach

This is a clean, modular pipeline for star detection and identification using a pairwise matching approach.

### **Requirements**
Install all dependencies with:
```bash
pip install -r requirements.txt
```

### **How to Run**
1. **Navigate to the `src` directory:**
   ```bash
   cd ~/SPARCS/src
   ```
2. **Run the main pipeline:**
   ```bash
   python3 -m pair_approach.main
   ```

- The script will:
  - Load the image at `images/MatchingImage.png`
  - Detect stars and convert them to vectors
  - Load and process the Hipparcos catalog
  - Match detected stars to catalog IDs
  - Print the results and save outputs in the `outputs/` folder

> **Note:** Ensure all required input files (e.g., `HipparcosCatalog.txt`, `images/MatchingImage.png`) are present in the correct directories. Output files (e.g., CSVs) will be generated in the `outputs/` directory.

---

## How to Run the Modular OpenCV Approach (Recommended for OpenCV-based Star Detection)

This approach uses a fully modular pipeline based on OpenCV for star detection and Wahba's method for attitude estimation. Each step is separated into its own module for clarity and maintainability.

### **Requirements**
Install all dependencies with:
```bash
pip install -r requirements.txt
```

### **How to Run**
1. **Navigate to the `other_approach` directory:**
   ```bash
   cd ~/SPARCS/src/other_approach
   ```
2. **Run the main pipeline:**
   ```bash
   python3 main.py
   ```

- The pipeline will:
  - Detect stars in the image using OpenCV and display the results
  - Convert detected centroids to unit vectors in the camera frame
  - Load the Hipparcos star catalog and convert catalog entries to unit vectors
  - Match detected stars to catalog stars based on angular distance
  - Print and save the matching results
  - If enough matches are found, compute and print the camera's attitude (rotation matrix) using Wahba's method

> **Note:** Ensure all required input files (e.g., `HipparcosCatalog.txt`, `images/MatchingImage.png`) are present in the correct directories. Output files (e.g., CSVs) will be generated in the `outputs/` directory.

## License

[MIT](https://choosealicense.com/licenses/mit/)
