import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
os.environ['TK_SILENCE_DEPRECATION'] = "1"

import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend

import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy.spatial.distance import cdist
import pandas as pd
from datetime import datetime




def get_shoreline_points(predict):
    with torch.no_grad():
        
        shoreline_map = predict.squeeze().cpu().numpy()  # Remove batch & channel dimensions

        # Threshold the output to get a binary map where 1=shoreline, 0=background
        shoreline_binary = (shoreline_map > 0.5).astype(np.uint8)

        # Get coordinates of all shoreline pixels
        shoreline_points = np.argwhere(shoreline_binary > 0)

        # If no shoreline points found, return empty array
        if len(shoreline_points) == 0:
            return np.array([])

        return shoreline_points


def order_shoreline_points(points, max_gap=55):
    """Order shoreline points to form a sequential line"""
    if len(points) == 0:
        return np.array([])

    # Start with the leftmost point (smallest x value)
    ordered = [points[np.argmin(points[:, 1])]]
    remaining = list(range(len(points)))
    remaining.remove(np.argmin(points[:, 1]))

    while remaining:
        last_point = ordered[-1]
        distances = np.sqrt(np.sum((points[remaining] - last_point)**2, axis=1))

        # Find closest point
        closest_idx = np.argmin(distances)
        closest_point_idx = remaining[closest_idx]

        # Check if distance is reasonable (to avoid jumps)
        if distances[closest_idx] > max_gap:
            # If no close points, try to find another segment
            if len(remaining) > len(points) / 2:  # If many points remain
                # Start a new segment with the leftmost remaining point
                leftmost_idx = np.argmin(points[remaining][:, 1])
                ordered.append(points[remaining[leftmost_idx]])
                remaining.pop(leftmost_idx)
            else:
                # If few points remain, we're probably done with the main shoreline
                break
        else:
            # Add closest point to ordered list
            ordered.append(points[closest_point_idx])
            remaining.pop(closest_idx)

    return np.array(ordered)

# Generate transects along a baseline
def generate_transects(baseline, num_transects=100, transect_length=100):
      
    indices = np.linspace(0, len(baseline)-1, num_transects).astype(int)
    points = baseline[indices]

    # Calculate normal vectors
    transects = []
    for i in range(len(points)):
        if i == 0:
            v = points[1] - points[0]
        elif i == len(points)-1:
            v = points[i] - points[i-1]
        else:
            v = points[i+1] - points[i-1]

        # Normalize
        v = v / np.linalg.norm(v)
        # Get perpendicular vector (clockwise rotation)
        normal = np.array([-v[1], v[0]])

        # Create transect line (extending in both directions)
        start_point = points[i] - normal * transect_length/2
        end_point = points[i] + normal * transect_length/2
        transects.append((start_point, end_point))

    return transects

# Find intersection points of a line with a mask
def find_intersection_points(line_start, line_end, mask):
        
    line_start_cv = (int(line_start[1]), int(line_start[0]))
    line_end_cv = (int(line_end[1]), int(line_end[0]))

    # Create a blank image
    blank = np.zeros_like(mask)

    # Draw the line
    cv2.line(blank, line_start_cv, line_end_cv, 1, 1)

    # Find intersection of line with mask
    intersection = np.logical_and(blank, mask).astype(np.uint8)

    # Get coordinates of intersection points
    points = np.argwhere(intersection > 0)

    if len(points) == 0:
        return None
    median_idx = len(points) // 2
    return np.array([points[median_idx][0], points[median_idx][1]])  # Return [y, x]

# Calculate NSM and EPR with signed values (positive for accretion, negative for erosion)
def calculate_shoreline_change(shoreline1_mask, shoreline2_mask, transects, pixel_to_meter=1.0, time_interval_years=1.0):
    nsm_values = []
    epr_values = []
    intersection_points1 = []
    intersection_points2 = []

    valid_transects = []

    for i, transect in enumerate(transects):
        # Find intersection points
        intersection1 = find_intersection_points(transect[0], transect[1], shoreline1_mask)
        intersection2 = find_intersection_points(transect[0], transect[1], shoreline2_mask)

        # Skip transects that don't intersect both shorelines
        if intersection1 is None or intersection2 is None:
            continue

        valid_transects.append(transect)
        intersection_points1.append(intersection1)
        intersection_points2.append(intersection2)

        # Calculate distance (NSM)
        distance = np.linalg.norm(intersection2 - intersection1) * pixel_to_meter

        
        # Get vector from land to water (transect direction)
        transect_vector = transect[1] - transect[0]  # Vector pointing from land to water
        transect_vector = transect_vector / np.linalg.norm(transect_vector)  # Normalize

        # Vector from shoreline1 to shoreline2
        movement_vector = intersection2 - intersection1

        direction = np.dot(movement_vector, transect_vector)
        sign = 1 if direction > 0 else -1

        # Apply sign to the distance
        signed_distance = sign * distance
        nsm_values.append(signed_distance)

        # Calculate EPR (inherits the sign from NSM)
        epr = signed_distance / time_interval_years
        epr_values.append(epr)

    return np.array(nsm_values), np.array(epr_values), np.array(intersection_points1), np.array(intersection_points2), valid_transects

# Modify this function to use a non-interactive backend 
def visualize_shoreline_change(image1, image2, shoreline1, shoreline2, transects,
                              intersection_points1, intersection_points2, nsm_values):
    """Visualize shoreline change analysis results"""
    import matplotlib
    matplotlib.use('Agg')  # Force non-interactive backend
    import matplotlib.pyplot as plt
    
    # Create a figure for the combined visualization
    fig = plt.figure(figsize=(12, 10))

    # Create masks for visualization
    mask1 = np.zeros((image1.shape[0], image1.shape[1]), dtype=np.uint8)
    mask2 = np.zeros((image2.shape[0], image2.shape[1]), dtype=np.uint8)

    # Draw shorelines on masks
    for point in shoreline1:
        mask1[point[0], point[1]] = 255
    for point in shoreline2:
        mask2[point[0], point[1]] = 255

    # Dilate masks to make shorelines more visible
    kernel = np.ones((3, 3), np.uint8)
    mask1 = cv2.dilate(mask1, kernel, iterations=1)
    mask2 = cv2.dilate(mask2, kernel, iterations=1)

    # Create a blank image to show both shorelines
    combined_mask = np.zeros((image1.shape[0], image1.shape[1], 3), dtype=np.uint8)
    combined_mask[..., 0] = mask1  # First shoreline in red channel
    combined_mask[..., 2] = mask2  # Second shoreline in blue channel

    # Display the combined mask
    plt.imshow(combined_mask)

    # Get color range for NSM values (separate for accretion and erosion)
    max_accretion = max(max(nsm_values), 0) if len(nsm_values) > 0 else 0  # Max positive value
    max_erosion = abs(min(min(nsm_values), 0)) if len(nsm_values) > 0 else 0  # Max negative value (as positive)

    # Plot transects with colors indicating accretion/erosion
    for i, (transect, p1, p2, nsm) in enumerate(zip(transects, intersection_points1, intersection_points2, nsm_values)):
        # Plot transect line in white
        plt.plot([transect[0][1], transect[1][1]], [transect[0][0], transect[1][0]], 'w-', alpha=0.3)

        # Plot intersection points
        plt.plot(p1[1], p1[0], 'yo', markersize=4)  # Yellow for first shoreline intersection
        plt.plot(p2[1], p2[0], 'co', markersize=4)  # Cyan for second shoreline intersection

        # Determine line width based on absolute NSM value
        abs_nsm = abs(nsm)
        max_val = max(max_accretion, max_erosion)
        normalized_width = 1 + 4 * (abs_nsm / max_val) if max_val > 0 else 1

        # Color based on accretion/erosion:
        # Green for accretion (positive NSM)
        # Red for erosion (negative NSM)
        if nsm > 0:
            plt.plot([p1[1], p2[1]], [p1[0], p2[0]], 'g-', linewidth=normalized_width)
        else:
            plt.plot([p1[1], p2[1]], [p1[0], p2[0]], 'r-', linewidth=normalized_width)

        # Only show label for some transects to avoid clutter
        if i % 5 == 0:  # Show every 5th transect label
            # Add NSM value text
            mid_x = (p1[1] + p2[1]) / 2
            mid_y = (p1[0] + p2[0]) / 2
            plt.text(mid_x, mid_y, f"{nsm:.1f}m", fontsize=8, ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.1'))

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], color='r', lw=2, label='Shoreline 1'),
        plt.Line2D([0], [0], color='b', lw=2, label='Shoreline 2'),
        plt.Line2D([0], [0], color='w', lw=1, alpha=0.3, label='Transect'),
        plt.Line2D([0], [0], color='g', lw=2, label='Accretion (+)'),
        plt.Line2D([0], [0], color='r', lw=2, label='Erosion (-)'),
        plt.Line2D([0], [0], marker='o', color='w', label='Intersection Points',
                  markerfacecolor='y', markersize=8)
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.title("Shoreline Change Analysis\nGreen = Accretion, Red = Erosion\nThicker lines indicate larger change")

    # Remove axes ticks for cleaner visualization
    plt.xticks([])
    plt.yticks([])

    # Save the figure
    os.makedirs("analysis_results", exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join("analysis_results", "shoreline_change.png"), dpi=300)
    plt.close(fig)  # Explicitly close the figure

    # Return statistics for further analysis
    stats = {
        "avg_nsm": np.mean(nsm_values) if len(nsm_values) > 0 else 0,
        "max_nsm": np.max(nsm_values) if len(nsm_values) > 0 else 0,
        "min_nsm": np.min(nsm_values) if len(nsm_values) > 0 else 0,
        "std_nsm": np.std(nsm_values) if len(nsm_values) > 0 else 0,
        "accretion_percent": np.sum(nsm_values > 0) / len(nsm_values) * 100 if len(nsm_values) > 0 else 0,
        "erosion_percent": np.sum(nsm_values < 0) / len(nsm_values) * 100 if len(nsm_values) > 0 else 0
    }

    # Display statistics
    # print(f"Average NSM: {stats['avg_nsm']:.2f} meters")
    # print(f"Maximum accretion: {stats['max_nsm']:.2f} meters")
    # print(f"Maximum erosion: {abs(stats['min_nsm']):.2f} meters")
    # print(f"Standard Deviation: {stats['std_nsm']:.2f} meters")
    # print(f"Accretion percentage: {stats['accretion_percent']:.1f}%")
    # print(f"Erosion percentage: {stats['erosion_percent']:.1f}%")

    return stats

# Main function to process satellite images and calculate NSM/EPR
def analyze_shoreline_change(mask1, mask2, date1=None, date2=None,
                              pixel_to_meter=10.0, num_transects=50):
    """
    Analyze shoreline change between two preprocessed masks.

    Parameters:
    -----------
    mask1, mask2 : np.ndarray
        Binary masks of the shorelines (1=shoreline, 0=background).
    date1, date2 : str
        Dates of the images in format 'YYYY-MM-DD', used to calculate EPR.
    pixel_to_meter : float
        Conversion from pixel distance to meters (depends on image resolution).
    num_transects : int
        Number of transects to generate for measurement.

    Returns:
    --------
    dict
        Dictionary containing NSM and EPR statistics.
    """

    # mask1 = cv2.imread(mask1, cv2.IMREAD_GRAYSCALE)
    # mask2 = cv2.imread(mask2, cv2.IMREAD_GRAYSCALE)
    
    if mask1 is None:
        print("Failed to load mask1 image.")
        # Handle the error, maybe skip processing or load a default mask
    else:
        shoreline_points1 = np.argwhere(mask1 > 0)


    # Parse dates if provided
    if date1 and date2:
        date1_obj = datetime.strptime(date1, '%Y-%m-%d')
        date2_obj = datetime.strptime(date2, '%Y-%m-%d')
        time_interval_years = (date2_obj - date1_obj).days / 365.25
    else:
        time_interval_years = 1.0  # Default to 1 year if dates not provided

    print("Processing first mask")
    # Extract shoreline points from the first mask
    shoreline_points1 = np.argwhere(mask1 > 0)
    shoreline1 = order_shoreline_points(shoreline_points1)

    # print(f"Extracted shoreline 1 with {len(shoreline1)} points")





    print("Processing second mask")
    # Extract shoreline points from the second mask
    shoreline_points2 = np.argwhere(mask2 > 0)
    shoreline2 = order_shoreline_points(shoreline_points2)

    # print(f"Extracted shoreline 2 with {len(shoreline2)} points")





    # Use the first shoreline as baseline for transects
    # print(f"Generating {num_transects} transects")
    transects = generate_transects(shoreline1, num_transects=num_transects,
                                   transect_length=min(mask1.shape[0], mask1.shape[1]) / 2)

    # Calculate NSM and EPR
    # print("Calculating NSM and EPR")
    nsm_values, epr_values, intersection_points1, intersection_points2, valid_transects = calculate_shoreline_change(
        mask1, mask2, transects,
        pixel_to_meter=pixel_to_meter,
        time_interval_years=time_interval_years
    )

    # Visualize results
    # print("Generating visualization")
    stats = visualize_shoreline_change(
        mask1, mask2, shoreline1, shoreline2, valid_transects,
        intersection_points1, intersection_points2,
        nsm_values
    )

    # Add EPR statistics
    if len(epr_values) > 0:
        stats["avg_epr"] = np.mean(epr_values)
        stats["max_epr"] = np.max(epr_values)
        stats["min_epr"] = np.min(epr_values)
        stats["std_epr"] = np.std(epr_values)

        # # Save detailed results to CSV
        # results_df = pd.DataFrame({
        #     'Transect': range(1, len(nsm_values) + 1),
        #     'NSM (m)': nsm_values,
        #     'EPR (m/year)': epr_values
        # })
        # results_df.to_csv("shoreline_change_results.csv", index=False)

        # print(f"Average EPR: {stats['avg_epr']:.2f} meters/year")
        # print(f"Results saved to shoreline_change_results.csv")
    else:
        print("WARNING: No valid transect intersections were found!")
        stats["avg_epr"] = 0
        stats["max_epr"] = 0
        stats["min_epr"] = 0
        stats["std_epr"] = 0
        # results_df = pd.DataFrame(columns=['Transect', 'NSM (m)', 'EPR (m/year)'])

    # return stats, results_df
    return stats



# Example usage
def run_shoreline_analysis(image1_path, image2_path, model_name):
    """Run shoreline analysis between two segmented images
    
    Args:
        image1_path: Path to first segmented image
        image2_path: Path to second segmented image
        model_name: Name of the model used for segmentation
        
    Returns:
        tuple: (EPR, NSM) values
    """
    # Import matplotlib with non-interactive backend to avoid tkinter issues
    import matplotlib
    matplotlib.use('Agg')  # Force non-interactive backend
    import matplotlib.pyplot as plt
    
    # Dates of the images (for EPR calculation)
    date1 = "2023-10-14"
    date2 = "2023-11-18"
    
    # Read images
    print(f"Reading segmented images from {model_name}")
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
    
    if image1 is None:
        raise ValueError(f"Could not read image at {image1_path}")
    if image2 is None:
        raise ValueError(f"Could not read image at {image2_path}")

    original_width, original_height = 1156, 1722  # Original image dimensions
    resized_size = 540  # Resized dimension

    # If original 10km x 10km map is 1156x1722 pixels:
    # Original resolution: 10000m / max(1156, 1722) ≈ 5.81m per pixel
    # After resizing to 540x540:
    # Original resolution: 10000m / max(1156, 1722) ≈ 5.81m per pixel
    # After resizing to 540x540:
    # New resolution = original_resolution * (original_size / resized_size)
    original_resolution = 10000 / max(original_width, original_height)  # meters per pixel in original
    pixel_to_meter = original_resolution * (max(original_width, original_height) / resized_size)

    # print(f"Original resolution: {original_resolution:.2f} m/pixel")
    # print(f"After resizing: {pixel_to_meter:.2f} m/pixel")

    # Create results directory if it doesn't exist
    os.makedirs("analysis_results", exist_ok=True)
    
    # Set the output filename based on model name
    plt.rcParams['figure.max_open_warning'] = 0  # Suppress max figure warning
    output_image = os.path.join("analysis_results", f"{model_name}_change_analysis.png")
    
    # Run the analysis
    try:
        stats = analyze_shoreline_change(
            image1, image2,
            date1=date1, date2=date2,
            pixel_to_meter=pixel_to_meter,
            num_transects=100
        )
        
        # Close all matplotlib figures to prevent memory leaks
        plt.close('all')
        
        # Return the average EPR and NSM values to be used by the Flask application
        return stats["avg_epr"], stats["avg_nsm"]
    except Exception as e:
        # Close all matplotlib figures on error too
        plt.close('all')
        raise e

if __name__ == "__main__":

    image1 = "DeepLab_segmented_1_sentinel2_void_2023-10-14_Weligama.jpg"
    image2 = "DeepLab_segmented_2_sentinel2_void_2023-11-18_Weligama.jpg"

    unet_epr, unet_nsm = run_shoreline_analysis(image1, image2, "U-Net")
    print(f"U-Net EPR: {unet_epr:.2f} m/year")
    print(f"U-Net NSM: {unet_nsm:.2f} m")
