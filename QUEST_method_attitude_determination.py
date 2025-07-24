import numpy as np
import pandas as pd
from scipy.linalg import eigh

class HipparcosCatalog:
    """Class to manage the Hipparcos catalog"""
    
    def __init__(self, catalog_file):
        self.catalog = self._load_catalog(catalog_file)
    
    def _load_catalog(self, catalog_file):
        """Load and process the Hipparcos catalog"""
        try:
            with open(catalog_file, 'r') as f:
                lines = f.readlines()
            
            print(f"Total lines in catalog: {len(lines)}")
            
            catalog_data = []
            for line in lines:
                # Skip header line and empty lines
                if 'name' in line or len(line.strip()) == 0:
                    continue
                    
                # Split by | and clean up
                if '|' in line:
                    parts = [part.strip() for part in line.split('|') if part.strip()]
                    
                    if len(parts) >= 3 and parts[0].startswith('HIP'):
                        try:
                            # Extract HIP ID
                            hip_id = int(parts[0].split()[1])
                            
                            # Parse RA (format: "06 45 09.2499")
                            ra_parts = parts[1].split()
                            ra_h = int(ra_parts[0])
                            ra_m = int(ra_parts[1])
                            ra_s = float(ra_parts[2])
                            ra_deg = (ra_h + ra_m/60.0 + ra_s/3600.0) * 15.0
                            
                            # Parse Dec (format: "-16 42 47.315")
                            dec_parts = parts[2].split()
                            dec_sign = 1 if dec_parts[0][0] != '-' else -1
                            dec_d = int(dec_parts[0][1:] if dec_parts[0][0] in ['+', '-'] else dec_parts[0])
                            dec_m = int(dec_parts[1])
                            dec_s = float(dec_parts[2])
                            dec_deg = dec_sign * (abs(dec_d) + dec_m/60.0 + dec_s/3600.0)
                            
                            catalog_data.append({
                                'HIP': hip_id,
                                'RA_deg': ra_deg,
                                'Dec_deg': dec_deg
                            })
                            
                        except (ValueError, IndexError) as e:
                            print(f"Error parsing line: {line.strip()}")
                            continue
            
            print(f"Successfully parsed {len(catalog_data)} stars")
            return pd.DataFrame(catalog_data)
        
        except Exception as e:
            print(f"Error loading catalog: {e}")
            return pd.DataFrame()
    
    def get_star_coords(self, hip_id):
        """Get RA/Dec coordinates of a star from the catalog"""
        star = self.catalog[self.catalog['HIP'] == hip_id]
        if len(star) == 0:
            return None, None
        return star.iloc[0]['RA_deg'], star.iloc[0]['Dec_deg']

class AttitudeDetermination:
    """Class for attitude calculation using the QUEST method"""
    
    @staticmethod
    def radec_to_unit_vector(ra_deg, dec_deg):
        """Convert RA/Dec coordinates to 3D unit vector"""
        ra_rad = np.radians(ra_deg)
        dec_rad = np.radians(dec_deg)
        
        x = np.cos(dec_rad) * np.cos(ra_rad)
        y = np.cos(dec_rad) * np.sin(ra_rad)
        z = np.sin(dec_rad)
        
        return np.array([x, y, z])
    
    @staticmethod
    def quest_algorithm(body_vectors, inertial_vectors):
        """Implement the QUEST algorithm for attitude determination"""
        n_stars = len(body_vectors)
        weights = np.ones(n_stars) / n_stars
        
        # Calculate B matrix
        B = np.zeros((3, 3))
        for i in range(n_stars):
            B += weights[i] * np.outer(body_vectors[i], inertial_vectors[i])
        
        # Calculate S = B + B^T and sigma = trace(B)
        S = B + B.T
        sigma = np.trace(B)
        
        # Calculate Z vector
        Z = np.array([
            B[1, 2] - B[2, 1],
            B[2, 0] - B[0, 2],
            B[0, 1] - B[1, 0]
        ])
        
        # Build K matrix for QUEST
        K = np.zeros((4, 4))
        K[0, 0] = sigma
        K[0, 1:4] = Z
        K[1:4, 0] = Z
        K[1:4, 1:4] = S - sigma * np.eye(3)
        
        # Find maximum eigenvalue and corresponding eigenvector
        eigenvalues, eigenvectors = eigh(K)
        max_idx = np.argmax(eigenvalues)
        quaternion = eigenvectors[:, max_idx]
        
        # Ensure q0 is positive
        if quaternion[0] < 0:
            quaternion = -quaternion
        
        # Convert quaternion to rotation matrix
        rotation_matrix = AttitudeDetermination.quaternion_to_rotation_matrix(quaternion)
        
        return quaternion, rotation_matrix
    
    @staticmethod
    def quaternion_to_rotation_matrix(q):
        """Convert a quaternion to rotation matrix"""
        q0, q1, q2, q3 = q
        
        R = np.array([
            [1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
            [2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)],
            [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2)]
        ])
        
        return R
    
    @staticmethod
    def rotation_matrix_to_euler(R):
        """Convert rotation matrix to Euler angles (roll, pitch, yaw)"""
        pitch = np.arcsin(-R[2, 0])
        
        if np.cos(pitch) > 1e-6:
            roll = np.arctan2(R[2, 1], R[2, 2])
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = 0
            yaw = np.arctan2(-R[0, 1], R[1, 1])
        
        return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)

def calculate_attitude(measurements_file, catalog_file):
    """
    Calculate attitude from star sensor measurements
    
    Args:
        measurements_file: file with centroids and HIP IDs (format: x y z HIP_ID)
        catalog_file: Hipparcos catalog file
    
    Returns:
        dict: attitude results
    """
    
    # Load catalog
    catalog = HipparcosCatalog(catalog_file)
    print(f"Catalog loaded with {len(catalog.catalog)} stars")
    
    # Load measurements
    measurements = pd.read_csv(measurements_file, sep='\t', 
                             names=['x', 'y', 'z', 'HIP_ID'], skiprows=0)
    
    # Check if first row contains headers and skip if necessary
    if measurements.iloc[0]['HIP_ID'] == 'HIP_ID' or isinstance(measurements.iloc[0]['HIP_ID'], str):
        measurements = pd.read_csv(measurements_file, sep='\t', 
                                 names=['x', 'y', 'z', 'HIP_ID'], skiprows=1)
    
    print(f"Loaded {len(measurements)} star measurements")
    
    # Prepare vectors for QUEST
    body_vectors = []
    inertial_vectors = []
    matched_stars = []
    
    for idx, row in measurements.iterrows():
        hip_id = int(row['HIP_ID'])
        ra, dec = catalog.get_star_coords(hip_id)
        
        if ra is not None and dec is not None:
            # Body frame vector (from sensor) - normalize
            body_vec = np.array([row['x'], row['y'], row['z']])
            body_vec = body_vec / np.linalg.norm(body_vec)
            
            # Inertial frame vector (from catalog)
            inertial_vec = AttitudeDetermination.radec_to_unit_vector(ra, dec)
            
            body_vectors.append(body_vec)
            inertial_vectors.append(inertial_vec)
            matched_stars.append(hip_id)
        else:
            print(f"Star HIP {hip_id} not found in catalog")
    
    if len(body_vectors) < 2:
        print("Error: need at least 2 stars to calculate attitude")
        return None
    
    body_vectors = np.array(body_vectors)
    inertial_vectors = np.array(inertial_vectors)
    
    print(f"Using {len(matched_stars)} stars: {matched_stars}")
    
    # Apply QUEST algorithm
    quaternion, rotation_matrix = AttitudeDetermination.quest_algorithm(
        body_vectors, inertial_vectors)
    
    # Calculate Euler angles
    roll, pitch, yaw = AttitudeDetermination.rotation_matrix_to_euler(rotation_matrix)
    
    # Calculate residual errors
    residual_errors = []
    for i in range(len(body_vectors)):
        rotated_inertial = rotation_matrix @ inertial_vectors[i]
        error = np.arccos(np.clip(np.dot(body_vectors[i], rotated_inertial), -1, 1))
        residual_errors.append(np.degrees(error))
    
    results = {
        'quaternion': quaternion,
        'rotation_matrix': rotation_matrix,
        'euler_angles': (roll, pitch, yaw),
        'matched_stars': matched_stars,
        'residual_errors': residual_errors,
        'mean_error': np.mean(residual_errors),
        'max_error': np.max(residual_errors)
    }
    
    return results

def print_results(results):
    """Print attitude results"""
    if results is None:
        return
    
    print("\n" + "="*50)
    print("ATTITUDE DETERMINATION RESULTS")
    print("="*50)
    
    q = results['quaternion']
    print(f"\nQuaternion [q0, q1, q2, q3]: [{q[0]:.6f}, {q[1]:.6f}, {q[2]:.6f}, {q[3]:.6f}]")
    
    roll, pitch, yaw = results['euler_angles']
    print(f"\nEuler Angles:")
    print(f"  Roll:  {roll:.3f}°")
    print(f"  Pitch: {pitch:.3f}°")
    print(f"  Yaw:   {yaw:.3f}°")
    
    print(f"\nRotation Matrix:")
    R = results['rotation_matrix']
    for i in range(3):
        print(f"  [{R[i,0]:8.5f} {R[i,1]:8.5f} {R[i,2]:8.5f}]")
    
    print(f"\nUsed {len(results['matched_stars'])} stars: {results['matched_stars']}")
    print(f"Mean error: {results['mean_error']:.4f}°")
    print(f"Max error: {results['max_error']:.4f}°")

# Main execution
if __name__ == "__main__":
    # File paths
    catalog_file = r"C:\Users\157205\Desktop\SPARCS-main\reducedHYp.txt"
    measurements_file = r"C:\Users\157205\Desktop\SPARCS-main\diomerdone.txt"
    
    print("STAR SENSOR ATTITUDE DETERMINATION")
    print("="*50)
    
    # Calculate attitude
    results = calculate_attitude(measurements_file, catalog_file)
    
    # Print results
    print_results(results)
    
    # Save results to file
    if results:
        with open('attitude_results.txt', 'w') as f:
            q = results['quaternion']
            f.write(f"Quaternion: [{q[0]:.6f}, {q[1]:.6f}, {q[2]:.6f}, {q[3]:.6f}]\n")
            roll, pitch, yaw = results['euler_angles']
            f.write(f"Roll: {roll:.3f}°, Pitch: {pitch:.3f}°, Yaw: {yaw:.3f}°\n")
            f.write(f"Mean error: {results['mean_error']:.4f}°\n")
        print("\nResults saved to 'attitude_results.txt'")