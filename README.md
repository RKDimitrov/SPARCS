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

## How to Run the Latest Demo
1. Power on the Raspberry Pi with the AI camera and BNO055 IMU connected.
2. Access the Raspberry Pi via SSH or connect directly with a monitor and keyboard.
3. Open a terminal and navigate to the project directory:
   
   Install the needed libraries:

   ```bash
    pip install numpy pandas matplotlib scipy
   ```
   Then:
   ```bash
   cd ~/SPARCS/src/code
   ```

   Then, run the following scripts in order:
   
   **a. Detect stars in an image:**
   ```bash
   python3 Detection.py
   ```
   - This script processes `MatchingImage.png` (or your own image) to detect star centroids and outputs detected star properties and vectors.
   
   **b. Try alternative detection:**
   ```bash
   python3 Z_new_Detection_trial.py
   ```
   - This script uses OpenCV for star detection and vector calculation as an alternative approach.
   
   **c. Generate catalog star triads or pairs:**
   ```bash
   python3 starRecognition.py
   python3 starRecogPairs.py
   ```
   - These scripts process the Hipparcos catalog to generate geometric properties for star triads and pairs, respectively, for later matching.
   
   **e. (OPTIONAL) Attitude determination (QUEST method):**
   ```bash
   python3 QUEST_method_attitude_determination.py
   ```
   - This script uses the matched stars to compute the spacecraft's attitude using the QUEST algorithm.
   
   > **Note:** Ensure all required input files (e.g., `HipparcosCatalog.txt`, `MatchingImage.png`) are present in the correct directories. Output files (e.g., CSVs) will be generated in the same or `outputs/` directory.

## License

[MIT](https://choosealicense.com/licenses/mit/)
