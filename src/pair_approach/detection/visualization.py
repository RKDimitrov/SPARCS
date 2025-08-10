from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def save_grayscale(img_pil, save_path='grayscale.png'):
    """Save a PIL grayscale image to disk."""
    img_pil.save(save_path, quality=95, optimize=True)
    print(f"Grayscale image saved to {save_path}")

def plot_detections(img, centroids, save_path='detected_stars.png'):
    """Plot detected star centroids on the image and save as PNG."""
    plt.figure(figsize=(12, 8))
    plt.imshow(img, cmap='gray')
    if len(centroids) > 0:
        plt.scatter(centroids[:, 1], centroids[:, 0], c='red', s=50, alpha=0.7, label='Detected Stars')
        
        # Add star numbers
        for i, (y, x) in enumerate(centroids):
            plt.annotate(f'{i+1}', (x, y), xytext=(5, 5), textcoords='offset points',
                        color='yellow', fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
    
    plt.legend()
    plt.title('Detected Stars')
    plt.axis('off')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"Detection plot saved to {save_path}")

def plot_matched_stars(img, centroids, matches, save_path='matched_stars.png'):
    """
    Plot detected stars with HIP catalog IDs for matched stars.
    
    Parameters:
    -----------
    img : numpy.ndarray
        The grayscale image
    centroids : numpy.ndarray
        Array of detected star centroids (y, x)
    matches : list
        List of match dictionaries with 'image_star_index' and 'catalog_star_id'
    save_path : str
        Path to save the output image
    """
    plt.figure(figsize=(16, 12))
    plt.imshow(img, cmap='gray', origin='upper')
    
    if len(centroids) == 0:
        plt.title('No Stars Detected')
        plt.axis('off')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black')
        plt.close()
        return
    
    # Create mapping from image star index to HIP ID
    match_map = {}
    if matches:
        for match in matches:
            img_idx = match['image_star_index']
            hip_id = match['catalog_star_id']
            match_map[img_idx] = hip_id
    
    # Plot all detected stars
    matched_indices = []
    unmatched_indices = []
    
    for i, (y, x) in enumerate(centroids):
        if i in match_map:
            matched_indices.append(i)
        else:
            unmatched_indices.append(i)
    
    # Plot matched stars (green with HIP IDs)
    if matched_indices:
        matched_centroids = centroids[matched_indices]
        plt.scatter(matched_centroids[:, 1], matched_centroids[:, 0], 
                   c='lime', s=80, alpha=0.8, label=f'Matched Stars ({len(matched_indices)})',
                   marker='o', edgecolors='white', linewidths=1)
        
        # Add HIP IDs for matched stars
        for i in matched_indices:
            y, x = centroids[i]
            hip_id = match_map[i]
            plt.annotate(f'HIP {hip_id}', (x, y), 
                        xytext=(10, 10), textcoords='offset points',
                        color='lime', fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8, edgecolor='lime'))
    
    # Plot unmatched stars (red with numbers)
    if unmatched_indices:
        unmatched_centroids = centroids[unmatched_indices]
        plt.scatter(unmatched_centroids[:, 1], unmatched_centroids[:, 0], 
                   c='red', s=60, alpha=0.7, label=f'Unmatched Stars ({len(unmatched_indices)})',
                   marker='x', linewidths=2)
        
        # Add star numbers for unmatched stars
        for i in unmatched_indices:
            y, x = centroids[i]
            plt.annotate(f'#{i+1}', (x, y), 
                        xytext=(5, -15), textcoords='offset points',
                        color='red', fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
    
    # Add title and legend
    total_stars = len(centroids)
    matched_count = len(matched_indices)
    plt.title(f'Star Identification Results: {matched_count}/{total_stars} stars matched', 
              color='white', fontsize=14, fontweight='bold', pad=20)
    
    plt.legend(loc='upper right', framealpha=0.9, facecolor='black', edgecolor='white')
    plt.axis('off')
    
    # Save with high quality
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black', edgecolor='none')
    plt.close()
    print(f"✅ Matched stars plot saved to {save_path}")
    print(f"   📊 {matched_count} stars matched with HIP catalog IDs")
    print(f"   📊 {len(unmatched_indices)} stars unmatched")

def plot_matching_comparison(img, centroids, initial_matches, final_matches, save_path='matching_comparison.png'):
    """
    Plot comparison between initial and final matching results.
    
    Parameters:
    -----------
    img : numpy.ndarray
        The grayscale image
    centroids : numpy.ndarray
        Array of detected star centroids (y, x)
    initial_matches : list
        Initial matches from pair matching
    final_matches : list
        Final matches after RANSAC
    save_path : str
        Path to save the output image
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Left plot: Initial matches
    ax1.imshow(img, cmap='gray', origin='upper')
    ax1.set_title('Initial Pair Matches', color='white', fontsize=12, fontweight='bold')
    
    if initial_matches:
        initial_map = {m['image_star_index']: m['catalog_star_id'] for m in initial_matches}
        for i, (y, x) in enumerate(centroids):
            if i in initial_map:
                ax1.scatter(x, y, c='orange', s=60, alpha=0.8, marker='o')
                ax1.annotate(f'HIP {initial_map[i]}', (x, y), 
                           xytext=(5, 5), textcoords='offset points',
                           color='orange', fontsize=8, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.8))
            else:
                ax1.scatter(x, y, c='gray', s=30, alpha=0.5, marker='x')
    
    ax1.axis('off')
    
    # Right plot: Final matches
    ax2.imshow(img, cmap='gray', origin='upper')
    ax2.set_title('Final RANSAC Matches', color='white', fontsize=12, fontweight='bold')
    
    if final_matches:
        final_map = {m['image_star_index']: m['catalog_star_id'] for m in final_matches}
        for i, (y, x) in enumerate(centroids):
            if i in final_map:
                # Color code by error if available
                color = 'lime'
                if 'angle_err_deg' in final_matches[0]:
                    error = next((m['angle_err_deg'] for m in final_matches if m['image_star_index'] == i), 0)
                    if error > 2.0:
                        color = 'yellow'
                    elif error > 1.0:
                        color = 'orange'
                
                ax2.scatter(x, y, c=color, s=60, alpha=0.8, marker='o')
                ax2.annotate(f'HIP {final_map[i]}', (x, y), 
                           xytext=(5, 5), textcoords='offset points',
                           color=color, fontsize=8, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.8))
            else:
                ax2.scatter(x, y, c='gray', s=30, alpha=0.5, marker='x')
    
    ax2.axis('off')
    
    # Overall title
    fig.suptitle(f'Matching Pipeline Results', color='white', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black', edgecolor='none')
    plt.close()
    print(f"✅ Matching comparison saved to {save_path}")

def create_star_catalog_overlay(img, centroids, matches, intensities=None, save_path='star_catalog_overlay.png'):
    """
    Create a detailed overlay with star catalog information.
    """
    plt.figure(figsize=(18, 12))
    plt.imshow(img, cmap='gray', origin='upper')
    
    if matches:
        match_map = {m['image_star_index']: m for m in matches}
        
        for i, (y, x) in enumerate(centroids):
            if i in match_map:
                match = match_map[i]
                hip_id = match['catalog_star_id']
                
                # Size based on intensity if available
                size = 80
                if intensities is not None and i < len(intensities):
                    # Scale size based on intensity (brightest = larger)
                    intensity_norm = intensities[i] / np.max(intensities)
                    size = 40 + 60 * intensity_norm
                
                # Plot star
                plt.scatter(x, y, c='cyan', s=size, alpha=0.7, 
                           marker='o', edgecolors='white', linewidths=1)
                
                # Error-based color coding if available
                error_text = ""
                text_color = 'cyan'
                if 'angle_err_deg' in match:
                    error = match['angle_err_deg']
                    error_text = f" ({error:.2f}°)"
                    if error > 2.0:
                        text_color = 'red'
                    elif error > 1.0:
                        text_color = 'yellow'
                    else:
                        text_color = 'lime'
                
                # Add detailed annotation
                plt.annotate(f'HIP {hip_id}{error_text}', (x, y), 
                           xytext=(15, 15), textcoords='offset points',
                           color=text_color, fontsize=9, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='black', 
                                   alpha=0.9, edgecolor=text_color),
                           arrowprops=dict(arrowstyle='->', color=text_color, alpha=0.7))
            else:
                # Unmatched star
                plt.scatter(x, y, c='red', s=40, alpha=0.6, marker='x', linewidths=2)
                plt.annotate(f'#{i+1}', (x, y), 
                           xytext=(5, -10), textcoords='offset points',
                           color='red', fontsize=7,
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
    
    plt.title('Star Catalog Identification with Error Analysis', 
              color='white', fontsize=16, fontweight='bold', pad=20)
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lime', markersize=10, 
                  label='Good Match (< 1°)', alpha=0.8),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=10, 
                  label='OK Match (1-2°)', alpha=0.8),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, 
                  label='Poor Match (> 2°)', alpha=0.8),
        plt.Line2D([0], [0], marker='x', color='red', markersize=8, 
                  label='Unmatched', alpha=0.8, linestyle='None')
    ]
    
    plt.legend(handles=legend_elements, loc='upper right', framealpha=0.9, 
              facecolor='black', edgecolor='white')
    
    plt.axis('off')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black', edgecolor='none')
    plt.close()
    print(f"✅ Detailed star catalog overlay saved to {save_path}")

def plot_final_result_summary(img, centroids, matches, quest_results=None, save_path='final_result_summary.png'):
    """
    Create a comprehensive summary plot with all results
    """
    fig = plt.figure(figsize=(20, 16))
    
    # Main image with matches
    ax_main = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
    ax_main.imshow(img, cmap='gray', origin='upper')
    
    if matches:
        match_map = {m['image_star_index']: m for m in matches}
        
        for i, (y, x) in enumerate(centroids):
            if i in match_map:
                match = match_map[i]
                hip_id = match['catalog_star_id']
                
                # Color by error if available
                color = 'lime'
                if 'angle_err_deg' in match:
                    error = match['angle_err_deg']
                    if error > 2.0:
                        color = 'red'
                    elif error > 1.0:
                        color = 'orange'
                
                ax_main.scatter(x, y, c=color, s=60, alpha=0.8, marker='o', edgecolors='white')
                ax_main.annotate(f'HIP {hip_id}', (x, y), 
                               xytext=(8, 8), textcoords='offset points',
                               color=color, fontsize=8, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.8))
            else:
                ax_main.scatter(x, y, c='gray', s=30, alpha=0.5, marker='x')
    
    ax_main.set_title('Final Star Tracker Results', color='white', fontsize=14, fontweight='bold')
    ax_main.axis('off')
    
    # Statistics panel
    ax_stats = plt.subplot2grid((3, 3), (0, 2))
    ax_stats.axis('off')
    
    stats_text = f"""DETECTION STATISTICS
    
Stars Detected: {len(centroids)}
Stars Matched: {len(matches) if matches else 0}
Match Rate: {len(matches)/len(centroids)*100:.1f}% 

ACCURACY METRICS"""
    
    if quest_results:
        stats_text += f"""
Mean Error: {quest_results.get('mean_error', 0):.2f}°
Max Error: {quest_results.get('max_error', 0):.2f}°
Solution Quality: {"EXCELLENT" if quest_results.get('mean_error', 999) < 2 else "GOOD" if quest_results.get('mean_error', 999) < 10 else "POOR"}"""
    
    ax_stats.text(0.1, 0.9, stats_text, transform=ax_stats.transAxes, 
                  color='white', fontsize=11, verticalalignment='top',
                  bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8))
    
    # Error histogram if available
    if matches and 'angle_err_deg' in matches[0]:
        ax_hist = plt.subplot2grid((3, 3), (1, 2))
        errors = [m['angle_err_deg'] for m in matches if 'angle_err_deg' in m]
        if errors:
            ax_hist.hist(errors, bins=10, color='cyan', alpha=0.7, edgecolor='white')
            ax_hist.set_xlabel('Error (degrees)', color='white')
            ax_hist.set_ylabel('Count', color='white')
            ax_hist.set_title('Error Distribution', color='white', fontsize=10)
            ax_hist.tick_params(colors='white')
            ax_hist.set_facecolor('black')
    
    # HIP ID list
    ax_list = plt.subplot2grid((3, 3), (2, 0), colspan=3)
    ax_list.axis('off')
    
    if matches:
        hip_ids = [str(m['catalog_star_id']) for m in matches]
        hip_text = "IDENTIFIED STARS (HIP IDs):\n" + ", ".join(hip_ids)
    else:
        hip_text = "No stars identified"
    
    ax_list.text(0.05, 0.95, hip_text, transform=ax_list.transAxes,
                 color='white', fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black', edgecolor='none')
    plt.close()
    print(f"✅ Final result summary saved to {save_path}")