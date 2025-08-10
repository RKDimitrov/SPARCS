import numpy as np

def centroids_to_vectors(centroids, img_height, img_width, fov_deg, 
                        fov_direction='horizontal', coordinate_system='standard'):
    """
    Convert centroids to 3D unit vectors given image size and field of view.
    
    Parameters:
    -----------
    fov_direction : str
        'horizontal', 'vertical', or 'diagonal' - what the fov_deg represents
    coordinate_system : str  
        'standard' (Y up), 'image' (Y down), 'flipped_x', 'flipped_y'
    """
    print(f"Converting {len(centroids)} centroids to vectors")
    print(f"Image size: {img_width}x{img_height}, FOV: {fov_deg}° ({fov_direction})")
    print(f"Coordinate system: {coordinate_system}")
    
    if len(centroids) == 0:
        return np.array([])
    
    fov_rad = np.deg2rad(fov_deg)
    
    # Calculate pixel scale based on FOV direction
    if fov_direction == 'horizontal':
        pixel_scale_x = fov_rad / img_width
        aspect_ratio = img_height / img_width
        pixel_scale_y = pixel_scale_x  # Square pixels
    elif fov_direction == 'vertical':
        pixel_scale_y = fov_rad / img_height  
        aspect_ratio = img_width / img_height
        pixel_scale_x = pixel_scale_y  # Square pixels
    elif fov_direction == 'diagonal':
        diagonal_pixels = np.sqrt(img_width**2 + img_height**2)
        pixel_scale = fov_rad / diagonal_pixels
        pixel_scale_x = pixel_scale_y = pixel_scale
    else:
        raise ValueError("fov_direction must be 'horizontal', 'vertical', or 'diagonal'")
    
    # Image center
    cy, cx = img_height / 2, img_width / 2
    
    vectors = []
    for i, (y, x) in enumerate(centroids):
        # Basic angular offsets
        dx = x - cx
        dy = y - cy
        
        # Apply coordinate system transformations
        if coordinate_system == 'standard':
            # Y up (standard astronomy)
            ang_x = dx * pixel_scale_x
            ang_y = -dy * pixel_scale_y  # Flip Y
        elif coordinate_system == 'image':
            # Y down (image coordinates)
            ang_x = dx * pixel_scale_x
            ang_y = dy * pixel_scale_y
        elif coordinate_system == 'flipped_x':
            # X flipped
            ang_x = -dx * pixel_scale_x
            ang_y = -dy * pixel_scale_y
        elif coordinate_system == 'flipped_y':
            # Y flipped differently
            ang_x = dx * pixel_scale_x
            ang_y = dy * pixel_scale_y
        else:
            raise ValueError("Unknown coordinate_system")
        
        # Pinhole camera model: [tan(θx), tan(θy), 1]
        vec = np.array([
            np.tan(ang_x),
            np.tan(ang_y), 
            1.0
        ])
        
        # Normalize to unit vector
        vec_normalized = vec / np.linalg.norm(vec)
        vectors.append(vec_normalized)
        
        # Debug first few stars
        if i < 3:
            print(f"  Star {i}: pixel=({x:.1f},{y:.1f}) → "
                  f"ang=({np.rad2deg(ang_x):.2f}°,{np.rad2deg(ang_y):.2f}°) → "
                  f"vec=({vec_normalized[0]:.3f},{vec_normalized[1]:.3f},{vec_normalized[2]:.3f})")
    
    return np.array(vectors)

def test_coordinate_systems(centroids, img_height, img_width, fov_deg):
    """Test all coordinate system variants"""
    print("\n=== TESTING COORDINATE SYSTEMS ===")
    
    systems = [
        ('horizontal', 'standard'),
        ('horizontal', 'image'), 
        ('horizontal', 'flipped_x'),
        ('horizontal', 'flipped_y'),
        ('vertical', 'standard'),
        ('diagonal', 'standard')
    ]
    
    results = {}
    for fov_dir, coord_sys in systems:
        print(f"\n--- Testing {fov_dir} FOV + {coord_sys} coordinates ---")
        try:
            vectors = centroids_to_vectors(centroids, img_height, img_width, fov_deg, 
                                         fov_dir, coord_sys)
            if len(vectors) > 0:
                max_angle = np.max([np.rad2deg(np.arccos(np.clip(v[2], -1, 1))) for v in vectors])
                results[f"{fov_dir}_{coord_sys}"] = {
                    'vectors': vectors,
                    'max_angle_from_center': max_angle
                }
                print(f"  Max angle from center: {max_angle:.1f}°")
            else:
                print("  No vectors generated")
        except Exception as e:
            print(f"  ERROR: {e}")
    
    return results