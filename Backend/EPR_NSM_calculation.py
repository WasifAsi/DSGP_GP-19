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



def sample_transect_points(image, point, normal, transect_length, is_vertical=False):
   
    h, w = image.shape[:2]
    samples = []
    print("sample transect points")
    # Adjust sampling strategy based on orientation
    if is_vertical:
        # For vertical shorelines, sample more horizontally and with finer steps
        steps = np.linspace(0.1, 1.0, 15)  # More concentrated samples
    else:
        steps = np.linspace(0.1, 1.0, 10)  # Standard sampling
        
    for step in steps:
        sample_point = point + normal * transect_length * step
        y = int(np.clip(sample_point[0], 0, h-1))
        x = int(np.clip(sample_point[1], 0, w-1))
        samples.append((y, x))
        
    return samples



def identify_sea_component(image, shoreline_points, transect_length=80):
   
    if len(shoreline_points) == 0:
        return np.array([0, 1])  # Default direction if no shoreline
        
    h, w = image.shape[:2]
    
    # Get the first and last points to determine overall orientation
    if len(shoreline_points) >= 10:
        first_point = shoreline_points[0]
        last_point = shoreline_points[-1]
        
        # Calculate overall direction and length
        shoreline_direction = last_point - first_point
        shoreline_length = np.linalg.norm(shoreline_direction)
        
        # Ensure we have a valid shoreline
        if shoreline_length < 10:  # Too short to be reliable
            return np.array([0, 1])  # Default direction
            
        # Normalize the direction vector
        shoreline_direction = shoreline_direction / shoreline_length
        
        # Detect if the shoreline is predominantly vertical
        # If the x component is small compared to y component, it's more vertical
        is_vertical = abs(shoreline_direction[1]) < abs(shoreline_direction[0] * 0.5)
        
        if is_vertical:
            print("DETECTED VERTICAL SHORELINE - Using specialized sampling")
        else:
            print("Detected horizontal shoreline - Using standard sampling")
            
        # Get the normal vectors (perpendicular to shoreline)
        normal_left = np.array([-shoreline_direction[1], shoreline_direction[0]])
        normal_right = -normal_left
        
        # Get a point in the middle of the shoreline for sampling
        middle_idx = len(shoreline_points) // 2
        middle_point = shoreline_points[middle_idx]
        
        # Create debug image for visualization
        debug_img = image.copy()
        
        # Draw shoreline on debug image
        for point in shoreline_points:
            y, x = int(point[0]), int(point[1])
            if 0 <= y < h and 0 <= x < w:
                cv2.circle(debug_img, (x, y), 1, (255, 255, 255), -1)
                
        # Draw normal vectors
        arrow_length = 40
        cv2.arrowedLine(debug_img,
                        (int(middle_point[1]), int(middle_point[0])),
                        (int(middle_point[1] + normal_left[1]*arrow_length),
                         int(middle_point[0] + normal_left[0]*arrow_length)),
                        (255, 0, 0), 2)
        cv2.arrowedLine(debug_img,
                        (int(middle_point[1]), int(middle_point[0])),
                        (int(middle_point[1] + normal_right[1]*arrow_length),
                         int(middle_point[0] + normal_right[0]*arrow_length)),
                        (0, 255, 0), 2)
                        
        # Add text labels for components
        cv2.putText(debug_img, "Component 1",
                   (int(middle_point[1] + normal_left[1]*arrow_length*1.2),
                    int(middle_point[0] + normal_left[0]*arrow_length*1.2)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(debug_img, "Component 2",
                   (int(middle_point[1] + normal_right[1]*arrow_length*1.2),
                    int(middle_point[0] + normal_right[0]*arrow_length*1.2)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                   
        # ---- COLOR-BASED ANALYSIS ----
        # Convert to HSV and create color masks
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Blue water mask (sea indicators)
        blue_mask = cv2.inRange(hsv_image, (90, 20, 30), (150, 255, 255))
        
        # Dark water mask (another sea indicator)
        dark_mask = cv2.inRange(hsv_image, (0, 0, 0), (180, 50, 100))
        
        # Green vegetation mask (land indicator)
        green_mask = cv2.inRange(hsv_image, (35, 30, 30), (85, 255, 255))
        
        # Brown/tan mask (sand/soil - land indicator)
        brown_mask = cv2.inRange(hsv_image, (10, 30, 80), (30, 200, 220))
        
        # Determine number of samples and sample range based on orientation
        if is_vertical:
            # For vertical shorelines, focus more on horizontal sampling
            left_sample_points = sample_transect_points(
                image, middle_point, normal_left, transect_length, is_vertical=True
            )
            right_sample_points = sample_transect_points(
                image, middle_point, normal_right, transect_length, is_vertical=True
            )
        else:
            # Standard sampling for horizontal shorelines
            left_sample_points = sample_transect_points(
                image, middle_point, normal_left, transect_length, is_vertical=False
            )
            right_sample_points = sample_transect_points(
                image, middle_point, normal_right, transect_length, is_vertical=False
            )
        
        # Initialize feature counts
        left_features = {"blue": 0, "dark": 0, "green": 0, "brown": 0}
        right_features = {"blue": 0, "dark": 0, "green": 0, "brown": 0}
        
        # Process left samples
        for y, x in left_sample_points:
            # Count features
            left_features["blue"] += blue_mask[y, x] > 0
            left_features["dark"] += dark_mask[y, x] > 0
            left_features["green"] += green_mask[y, x] > 0
            left_features["brown"] += brown_mask[y, x] > 0
            
            # Mark sample point on debug image
            color = (0, 0, 255) if blue_mask[y, x] > 0 else \
                   (0, 255, 0) if green_mask[y, x] > 0 else \
                   (165, 42, 42) if brown_mask[y, x] > 0 else \
                   (211, 211, 211)  # Default color
            cv2.circle(debug_img, (x, y), 3, color, -1)
            
        # Process right samples
        for y, x in right_sample_points:
            # Count features
            right_features["blue"] += blue_mask[y, x] > 0
            right_features["dark"] += dark_mask[y, x] > 0
            right_features["green"] += green_mask[y, x] > 0
            right_features["brown"] += brown_mask[y, x] > 0
            
            # Mark sample point on debug image
            color = (0, 0, 255) if blue_mask[y, x] > 0 else \
                   (0, 255, 0) if green_mask[y, x] > 0 else \
                   (165, 42, 42) if brown_mask[y, x] > 0 else \
                   (211, 211, 211)  # Default color
            cv2.circle(debug_img, (x, y), 3, color, -1)
            
        # Save debug image
        cv2.imwrite("component_analysis_improved.jpg", cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
        
        # Calculate sea scores with appropriate weighting based on orientation
        if is_vertical:
            # For vertical shorelines, emphasize water indicators more
            left_sea_score = (left_features["blue"] * 3) + (left_features["dark"] * 2) - \
                             (left_features["green"] * 1.5) - (left_features["brown"] * 1)
                             
            right_sea_score = (right_features["blue"] * 3) + (right_features["dark"] * 2) - \
                              (right_features["green"] * 1.5) - (right_features["brown"] * 1)
        else:
            # Standard weights for horizontal shorelines
            left_sea_score = (left_features["blue"] * 2) + (left_features["dark"] * 1) - \
                             (left_features["green"] * 2) - (left_features["brown"] * 3)
                             
            right_sea_score = (right_features["blue"] * 2) + (right_features["dark"] * 1) - \
                              (right_features["green"] * 2) - (right_features["brown"] * 3)
                              
        # Print debug information
        print("--- Orientation-Aware Component Analysis ---")
        print(f"Shoreline orientation: {'Vertical' if is_vertical else 'Horizontal'}")
        print(f"Component 1 (Left) - Blue: {left_features['blue']}, Dark: {left_features['dark']}, " +
              f"Green: {left_features['green']}, Brown: {left_features['brown']}")
        print(f"Component 2 (Right) - Blue: {right_features['blue']}, Dark: {right_features['dark']}, " +
              f"Green: {right_features['green']}, Brown: {right_features['brown']}")
        print(f"Component 1 Sea Score: {left_sea_score}")
        print(f"Component 2 Sea Score: {right_sea_score}")
        
        # ---- TEXTURE ANALYSIS ----
        # Sea tends to be more homogeneous (lower variance) than land
        left_variance = 0
        right_variance = 0
        
        # Reduced weight for texture analysis in vertical shorelines
        texture_weight = 0.5 if is_vertical else 1.0
        
        # Convert to grayscale for texture analysis
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Sample points for texture analysis (use fewer points for efficiency)
        left_texture_samples = left_sample_points[::3]  # Every 3rd point
        right_texture_samples = right_sample_points[::3]  # Every 3rd point
        
        # Process left samples
        for y, x in left_texture_samples:
            # Ensure within bounds for patch extraction
            if y >= 5 and y < h-5 and x >= 5 and x < w-5:
                # Get 5x5 patch around point
                patch = gray_image[y-5:y+6, x-5:x+6]
                left_variance += np.var(patch)
                
        # Process right samples
        for y, x in right_texture_samples:
            # Ensure within bounds for patch extraction
            if y >= 5 and y < h-5 and x >= 5 and x < w-5:
                # Get 5x5 patch around point
                patch = gray_image[y-5:y+6, x-5:x+6]
                right_variance += np.var(patch)
                
        # Normalize by number of samples
        left_variance /= len(left_texture_samples) if left_texture_samples else 1
        right_variance /= len(right_texture_samples) if right_texture_samples else 1
        
        print("--- Texture Analysis ---")
        print(f"Component 1 Variance: {left_variance:.2f}")
        print(f"Component 2 Variance: {right_variance:.2f}")
        
        # ---- COMBINE METHODS FOR FINAL DECISION ----
        # Initialize vote counters
        left_sea_votes = 0
        right_sea_votes = 0
        
        # Method 1: Color-based (highest weight)
        if left_sea_score > right_sea_score:
            left_sea_votes += 3
        else:
            right_sea_votes += 3
            
        # Method 2: Texture (sea has lower variance) with adjusted weight
        if left_variance < right_variance:
            left_sea_votes += texture_weight
        else:
            right_sea_votes += texture_weight
            
        # Add additional vote weight for vertical shorelines based on absolute blue counts
        if is_vertical:
            # Give extra weight to pure blue count
            if left_features["blue"] > right_features["blue"]:
                left_sea_votes += 2
            else:
                right_sea_votes += 2
                
        print("--- Final Voting Results ---")
        print(f"Component 1 sea votes: {left_sea_votes}")
        print(f"Component 2 sea votes: {right_sea_votes}")
        
        # Determine final sea direction
        if left_sea_votes >= right_sea_votes:
            print("DECISION: Component 1 (LEFT) is SEA")
            sea_direction = normal_left
        else:
            print("DECISION: Component 2 (RIGHT) is SEA")
            sea_direction = normal_right
            
        # Create final visualization
        final_img = image.copy()
        
        # Draw shoreline
        for point in shoreline_points:
            y, x = int(point[0]), int(point[1])
            if 0 <= y < h and 0 <= x < w:
                cv2.circle(final_img, (x, y), 1, (255, 255, 255), -1)
                
        # Draw sea direction arrow
        arrow_start = middle_point
        arrow_end = middle_point + sea_direction * 50
        
        cv2.arrowedLine(final_img,
                        (int(arrow_start[1]), int(arrow_start[0])),
                        (int(arrow_end[1]), int(arrow_end[0])),
                        (0, 255, 255), 3, tipLength=0.3)
                        
        # Add SEA and LAND labels
        sea_label_pos = middle_point + sea_direction * 70
        land_label_pos = middle_point - sea_direction * 70
        
        # Draw text backgrounds to make labels more visible
        cv2.putText(final_img, "SEA",
                   (int(sea_label_pos[1])-2, int(sea_label_pos[0])+2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
        cv2.putText(final_img, "LAND",
                   (int(land_label_pos[1])-2, int(land_label_pos[0])+2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
                   
        # Draw text
        cv2.putText(final_img, "SEA",
                   (int(sea_label_pos[1]), int(sea_label_pos[0])),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(final_img, "LAND",
                   (int(land_label_pos[1]), int(land_label_pos[0])),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                   
        # Save final visualization with orientation indicator
        orientation_text = "VERTICAL SHORELINE" if is_vertical else "HORIZONTAL SHORELINE"
        cv2.putText(final_img, orientation_text,
                   (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                   
        cv2.imwrite("sea_direction_result_improved.jpg", cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))
        
        return sea_direction
        
    # Fallback default direction
    return np.array([0, 1])


def create_landward_baseline(shoreline_points, sea_direction, offset_distance=50, image_shape=None):
   
    import numpy as np
    
    # Ensure we have enough points for a meaningful baseline
    if len(shoreline_points) < 5:
        return np.array([])
    
    # Calculate land direction (opposite of sea direction)
    land_direction = -sea_direction
    
    # Create baseline by offsetting shoreline points landward
    baseline_points = []
    
    # Use regular intervals to create a smoother baseline
    # This helps avoid duplicating the complexity of the shoreline
    num_baseline_points = min(100, len(shoreline_points))  # Limit number of points
    indices = np.linspace(0, len(shoreline_points)-1, num_baseline_points).astype(int)
    
    for idx in indices:
        shoreline_point = shoreline_points[idx]
        
        # Offset point landward
        baseline_point = shoreline_point + land_direction * offset_distance
        
        # Ensure baseline point stays within image boundaries if image_shape is provided
        #if image_shape is not None:
            #h, w = image_shape
            #baseline_point[0] = np.clip(baseline_point[0], 5, h-5)  # Keep 5 pixels from edge
            #baseline_point[1] = np.clip(baseline_point[1], 5, w-5)  # Keep 5 pixels from edge
            
        baseline_points.append(baseline_point)
    
    # Apply stronger smoothing to create a single clean line
    # Use Savitzky-Golay filter for better smoothing while preserving shape
    try:
        from scipy.signal import savgol_filter
        
        # Extract y and x coordinates
        y_coords = np.array([p[0] for p in baseline_points])
        x_coords = np.array([p[1] for p in baseline_points])
        
        # Apply Savitzky-Golay filter (window size 11, polynomial order 3)
        window_length = min(11, len(baseline_points) - 2)  # Must be odd and < len(points)
        if window_length % 2 == 0:  # Ensure window length is odd
            window_length -= 1
            
        if window_length > 3:  # Minimum required points for filter
            y_smooth = savgol_filter(y_coords, window_length, 3)
            x_smooth = savgol_filter(x_coords, window_length, 3)
            
            smoothed_baseline = np.column_stack((y_smooth, x_smooth))
        else:
            # Fallback to moving average for very short baselines
            smoothed_baseline = np.array(baseline_points)
    except ImportError:
        # Fallback if scipy is not available
        # Simple moving average smoothing with larger window
        window_size = min(11, len(baseline_points))
        smoothed_baseline = []
        
        for i in range(len(baseline_points)):
            # Get window indices with smooth handling for endpoints
            start_idx = max(0, i - window_size//2)
            end_idx = min(len(baseline_points), i + window_size//2 + 1)
            
            # Calculate average position within window
            window_points = np.array(baseline_points[start_idx:end_idx])
            avg_point = np.mean(window_points, axis=0)
            
            smoothed_baseline.append(avg_point)
            
        smoothed_baseline = np.array(smoothed_baseline)
    
    # Final check to ensure all points are within image boundaries
    #if image_shape is not None:
        #h, w = image_shape
        #smoothed_baseline[:, 0] = np.clip(smoothed_baseline[:, 0], 5, h-5)
        #smoothed_baseline[:, 1] = np.clip(smoothed_baseline[:, 1], 5, w-5)
    
    return smoothed_baseline


def generate_transects_from_baseline(baseline_points, sea_direction, num_transects=100, transect_length=100):
   
    import numpy as np
    
    # Ensure we have enough baseline points
    if len(baseline_points) < 3:
        return []
    
    # Sample points along the baseline
    indices = np.linspace(0, len(baseline_points)-1, num_transects).astype(int)
    points = baseline_points[indices]
    
    # Generate transects
    transects = []
    
    for i in range(len(points)):
        # Calculate local tangent to baseline
        if i == 0:
            # First point
            tangent = baseline_points[1] - baseline_points[0]
        elif i == len(baseline_points)-1:
            # Last point
            tangent = baseline_points[-1] - baseline_points[-2]
        else:
            # Middle points - use neighbors
            tangent = baseline_points[indices[i]+1] - baseline_points[indices[i]-1]
        
        # Normalize tangent
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm > 0:
            tangent = tangent / tangent_norm
        else:
            # Fallback if normalization fails
            tangent = np.array([0, 1])
        
        # Calculate normal (perpendicular to tangent)
        normal = np.array([-tangent[1], tangent[0]])
        
        # Ensure normal points toward sea by comparing with sea_direction
        if np.dot(normal, sea_direction) < 0:
            normal = -normal
        
        # Create transect (from baseline to seaward)
        start_point = points[i]  # Start at baseline point
        end_point = start_point + normal * transect_length
        
        transects.append((start_point, end_point))
    
    return transects

def visualize_baseline_and_transects(image, shoreline_points, baseline_points, transects, sea_direction):
   
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # ----------------
    # Create mask for shoreline
    shoreline_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    for point in shoreline_points:
        y, x = int(point[0]), int(point[1])
        if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
            shoreline_mask[y, x] = 1
    
    # Create mask for baseline with line drawing to ensure continuity
    baseline_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    
    # First, plot points
    for point in baseline_points:
        y, x = int(point[0]), int(point[1])
        if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
            baseline_mask[y, x] = 1
    
    # Second, connect adjacent points with lines to ensure continuity
    for i in range(len(baseline_points)-1):
        pt1 = (int(baseline_points[i][1]), int(baseline_points[i][0]))
        pt2 = (int(baseline_points[i+1][1]), int(baseline_points[i+1][0]))
        cv2.line(baseline_mask, pt1, pt2, 1, 1)  # Draw line between points
    
    # Create color overlay
    overlay = image.copy()
    
    # Add shoreline (blue)
    overlay[shoreline_mask > 0] = [0, 0, 255]  # Blue for shoreline
    
    # Add baseline (red) 
    overlay[baseline_mask > 0] = [255, 0, 0]  # Red for baseline
    
    # Blend with original image for better visibility
    alpha = 0.7
    result = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
    
    # Display the image with colored masks
    plt.imshow(result)
    # ----------------
    
    # Plot transects with reduced opacity to avoid cluttering
    max_display_transects = 100
    step = max(1, len(transects) // max_display_transects)
    
    for i in range(0, len(transects), step):
        start_point, end_point = transects[i]
        plt.plot([start_point[1], end_point[1]], [start_point[0], end_point[0]], 
                 'y-', linewidth=1, alpha=0.6)
    
    # Add sea direction indicator at center point
    if len(shoreline_points) > 0:
        center_idx = len(shoreline_points) // 2
        center_point = shoreline_points[center_idx]
        
        # Draw arrow pointing to sea
        plt.arrow(center_point[1], center_point[0],
                  sea_direction[1]*30, sea_direction[0]*30,
                  color='cyan', width=2, head_width=10, head_length=10)
        
        # Ensure label positions are within image boundaries
        h, w = image.shape[:2]
        
        # Add SEA label
        sea_label_point = center_point + sea_direction * 70
        sea_y = int(np.clip(sea_label_point[0], 30, h-30))
        sea_x = int(np.clip(sea_label_point[1], 50, w-50))
        
        plt.text(sea_x, sea_y, "SEA",
                 color='cyan', fontsize=14, fontweight='bold',
                 ha='center', va='center',
                 bbox=dict(facecolor='black', alpha=0.7, edgecolor='none'))
        
        # Add LAND label
        land_label_point = center_point - sea_direction * 70
        land_y = int(np.clip(land_label_point[0], 30, h-30))
        land_x = int(np.clip(land_label_point[1], 50, w-50))
        
        plt.text(land_x, land_y, "LAND",
                 color='green', fontsize=14, fontweight='bold',
                 ha='center', va='center',
                 bbox=dict(facecolor='black', alpha=0.7, edgecolor='none'))
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color='blue', lw=2, label='Shoreline'),
        plt.Line2D([0], [0], color='red', lw=2, label='Landward Baseline'),
        plt.Line2D([0], [0], color='yellow', lw=1, alpha=0.6, label='Transect')
    ]
    plt.legend(handles=legend_elements, loc='upper right', framealpha=0.8, facecolor='white')
    
    # Add title with background for visibility
    plt.title("Shoreline Analysis with Landward Baseline and Transects", 
              bbox=dict(facecolor='white', alpha=0.7))
    
    # Remove axes for cleaner visualization
    plt.xticks([])
    plt.yticks([])
    
    # Save figure
    plt.tight_layout()
    plt.savefig("baseline_and_transects.png", dpi=300)
    
    # Create OpenCV version for comparison
    cv_image = image.copy()
    
    # Create separate RGB image with colored masks for precise control
    colored_mask = np.zeros_like(cv_image)
    colored_mask[shoreline_mask > 0] = [255, 0, 0]  # BGR format - Blue for shoreline
    colored_mask[baseline_mask > 0] = [0, 0, 255]  # BGR format - Red for baseline
    
    # Blend with original image
    cv_image = cv2.addWeighted(cv_image, 0.7, colored_mask, 0.8, 0)
    
    # Draw transects
    for i in range(0, len(transects), step):
        start = (int(transects[i][0][1]), int(transects[i][0][0]))
        end = (int(transects[i][1][1]), int(transects[i][1][0]))
        cv2.line(cv_image, start, end, (0, 255, 255), 1)  # Yellow
    
    # Add direction arrow
    if len(shoreline_points) > 0:
        start = (int(center_point[1]), int(center_point[0]))
        end = (int(center_point[1] + sea_direction[1]*30), 
               int(center_point[0] + sea_direction[0]*30))
        cv2.arrowedLine(cv_image, start, end, (255, 255, 0), 2, tipLength=0.3)
        
        # Add labels
        cv2.putText(cv_image, "SEA", (sea_x, sea_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(cv_image, "SEA", (sea_x, sea_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    
        cv2.putText(cv_image, "LAND", (land_x, land_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(cv_image, "LAND", (land_x, land_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Add legend
    cv2.putText(cv_image, "Blue: Shoreline", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(cv_image, "Red: Landward Baseline", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Save OpenCV version
    cv2.imwrite("baseline_and_transects_cv.jpg", cv_image)
    
    plt.close()
    
    print("Visualizations saved with connected baseline lines")



def calculate_shoreline_change_with_direction(shoreline1_mask, shoreline2_mask, transects,
                                             pixel_to_meter=1.0, time_interval_years=1.0):
  
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

        # Get vector from land to sea (transect direction is already oriented toward sea)
        transect_vector = transect[1] - transect[0]
        transect_vector = transect_vector / np.linalg.norm(transect_vector)  # Normalize

        # Vector from shoreline1 to shoreline2
        movement_vector = intersection2 - intersection1

        # Check if movement aligns with transect direction (dot product)
        # If positive, shoreline moved toward sea (accretion)
        # If negative, shoreline moved toward land (erosion)
        direction = np.dot(movement_vector, transect_vector)
        sign = 1 if direction > 0 else -1

        # Apply sign to the distance
        signed_distance = sign * distance
        nsm_values.append(signed_distance)

        # Calculate EPR (inherits the sign from NSM)
        epr = signed_distance / time_interval_years
        epr_values.append(epr)

    return np.array(nsm_values), np.array(epr_values), np.array(intersection_points1), np.array(intersection_points2), valid_transects








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


# Modify this function to use a non-interactive backend 
def visualize_shoreline_change(image1, image2, model_name, shoreline1, shoreline2, transects,
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
    plt.savefig(os.path.join("analysis_results", f"{model_name}_shoreline_change.png"), dpi=300)
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
def analyze_shoreline_change(mask1, mask2,satelite1, satelite2, model_name,  date1=None, date2=None,
                              pixel_to_meter=10.0, num_transects=50, baseline_offset=50):
    """
    Analyze shoreline change between two preprocessed masks.

    Parameters:
    -----------
    mask1, mask2 : np.ndarray
        Binary masks of the shorelines (1=shoreline, 0=background).
    model_name : str
        Name of the model used for segmentation
    satelite1, satelite2 : np.ndarray
        Original satellite images for visualization and sea direction detection.
    date1, date2 : str
        Dates of the images in format 'YYYY-MM-DD', used to calculate EPR.
    pixel_to_meter : float
        Conversion from pixel distance to meters (depends on image resolution).
    num_transects : int
        Number of transects to generate for measurement.
    baseline_offset : int
        Distance to offset the baseline from the shoreline towards land.

    Returns:
    --------
    dict
        Dictionary containing NSM and EPR statistics.
    """
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

    print("Processing second mask")
    # Extract shoreline points from the second mask
    shoreline_points2 = np.argwhere(mask2 > 0)
    shoreline2 = order_shoreline_points(shoreline_points2)

    print("Detecting sea direction using robust component analysis")
    sea_direction = identify_sea_component(satelite1, shoreline1)

    print(f"Creating landward baseline offset by {baseline_offset} pixels")
    baseline = create_landward_baseline(
        shoreline1, 
        sea_direction, 
        offset_distance=baseline_offset,
        image_shape=mask1.shape[:2]  # Use mask1 instead of image1
    )

    # Ensure baseline has enough points
    if len(baseline) < 5:
        print("WARNING: Created baseline too short. Increasing number of baseline points.")
        # Try again with more points
        baseline = create_landward_baseline(
            shoreline1, 
            sea_direction, 
            offset_distance=baseline_offset,
            image_shape=mask1.shape[:2]  # Use mask1 instead of image1
        )
    
    # Generate transects from the landward baseline
    print(f"Generating {num_transects} transects from landward baseline")
    transects = generate_transects_from_baseline(
        baseline, 
        sea_direction,
        num_transects=num_transects,
        transect_length=min(mask1.shape[0], mask1.shape[1])/3  # Use mask1 instead of image1
    )

    # Visualize the baseline and transects
    visualize_baseline_and_transects(satelite1, shoreline1, baseline, transects, sea_direction)

    # Calculate NSM and EPR with the new transects
    print("Calculating NSM and EPR with transects from landward baseline")
    nsm_values, epr_values, intersection_points1, intersection_points2, valid_transects = calculate_shoreline_change_with_direction(
        mask1, mask2, transects,
        pixel_to_meter=pixel_to_meter,
        time_interval_years=time_interval_years
    )

    # Visualize results
    print("Generating change visualization")
    stats = visualize_shoreline_change(
        satelite1, satelite2, model_name, shoreline1, shoreline2, valid_transects,
        intersection_points1, intersection_points2, nsm_values
    )

    # Add EPR statistics
    if len(epr_values) > 0:
        stats["avg_epr"] = np.mean(epr_values)
        stats["max_epr"] = np.max(epr_values)
        stats["min_epr"] = np.min(epr_values)
        stats["std_epr"] = np.std(epr_values)
        
        # Save detailed results to CSV
        results_df = pd.DataFrame({
            'Transect': range(1, len(nsm_values) + 1),
            'NSM (m)': nsm_values,
            'EPR (m/year)': epr_values
        })
        os.makedirs("analysis_results", exist_ok=True)
        results_df.to_csv(os.path.join("analysis_results", f"{model_name}_shoreline_results.csv"), index=False)
        
        print(f"Average EPR: {stats['avg_epr']:.2f} meters/year")
        print(f"Results saved to shoreline_change_results_landward_baseline.csv")
    else:
        print("WARNING: No valid transect intersections were found!")
        stats["avg_epr"] = 0
        stats["max_epr"] = 0
        stats["min_epr"] = 0
        stats["std_epr"] = 0
        results_df = pd.DataFrame(columns=['Transect', 'NSM (m)', 'EPR (m/year)'])
    
    return stats



# Example usage
def run_shoreline_analysis(image1_path, image2_path, satelite1, satelite2, model_name):
    """Run shoreline analysis between two segmented images"""
    # Import required libraries
    import matplotlib
    matplotlib.use('Agg')  # Force non-interactive backend
    import matplotlib.pyplot as plt
    import shutil  # For file operations
    
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

    # Calculate resolution factors
    original_resolution = 10000 / max(original_width, original_height)  # meters per pixel in original
    pixel_to_meter = original_resolution * (max(original_width, original_height) / resized_size)

    # Create model-specific directory
    model_dir = os.path.join("analysis_results", model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Set the output filename based on model name
    plt.rcParams['figure.max_open_warning'] = 0  # Suppress max figure warning
    
    # Override the identify_sea_component function to save directly to model directory
    def model_identify_sea_component(image, shoreline_points, transect_length=80):
        """A full implementation that saves files to the model directory"""
        if len(shoreline_points) == 0:
            return np.array([0, 1])  # Default direction if no shoreline
        
        h, w = image.shape[:2]
        
        # Get the first and last points to determine overall orientation
        if len(shoreline_points) >= 10:
            first_point = shoreline_points[0]
            last_point = shoreline_points[-1]
            
            # Calculate overall direction and length
            shoreline_direction = last_point - first_point
            shoreline_length = np.linalg.norm(shoreline_direction)
            
            # Ensure we have a valid shoreline
            if shoreline_length < 10:  # Too short to be reliable
                return np.array([0, 1])  # Default direction
                
            # Normalize the direction vector
            shoreline_direction = shoreline_direction / shoreline_length
            
            # Detect if the shoreline is predominantly vertical
            is_vertical = abs(shoreline_direction[1]) < abs(shoreline_direction[0] * 0.5)
            
            if is_vertical:
                print("DETECTED VERTICAL SHORELINE - Using specialized sampling")
            else:
                print("Detected horizontal shoreline - Using standard sampling")
                
            # Get the normal vectors (perpendicular to shoreline)
            normal_left = np.array([-shoreline_direction[1], shoreline_direction[0]])
            normal_right = -normal_left
            
            # Get a point in the middle of the shoreline for sampling
            middle_idx = len(shoreline_points) // 2
            middle_point = shoreline_points[middle_idx]
            
            # Create debug image for visualization
            debug_img = image.copy()
            
            # Draw shoreline on debug image
            for point in shoreline_points:
                y, x = int(point[0]), int(point[1])
                if 0 <= y < h and 0 <= x < w:
                    cv2.circle(debug_img, (x, y), 1, (255, 255, 255), -1)
                    
            # Draw normal vectors
            arrow_length = 40
            cv2.arrowedLine(debug_img,
                            (int(middle_point[1]), int(middle_point[0])),
                            (int(middle_point[1] + normal_left[1]*arrow_length),
                             int(middle_point[0] + normal_left[0]*arrow_length)),
                            (255, 0, 0), 2)
            cv2.arrowedLine(debug_img,
                            (int(middle_point[1]), int(middle_point[0])),
                            (int(middle_point[1] + normal_right[1]*arrow_length),
                             int(middle_point[0] + normal_right[0]*arrow_length)),
                            (0, 255, 0), 2)
                            
            # Add text labels for components
            cv2.putText(debug_img, "Component 1",
                       (int(middle_point[1] + normal_left[1]*arrow_length*1.2),
                        int(middle_point[0] + normal_left[0]*arrow_length*1.2)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(debug_img, "Component 2",
                       (int(middle_point[1] + normal_right[1]*arrow_length*1.2),
                        int(middle_point[0] + normal_right[0]*arrow_length*1.2)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                       
            # ---- COLOR-BASED ANALYSIS ----
            hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # Blue water mask (sea indicators)
            blue_mask = cv2.inRange(hsv_image, (90, 20, 30), (150, 255, 255))
            
            # Dark water mask (another sea indicator)
            dark_mask = cv2.inRange(hsv_image, (0, 0, 0), (180, 50, 100))
            
            # Green vegetation mask (land indicator)
            green_mask = cv2.inRange(hsv_image, (35, 30, 30), (85, 255, 255))
            
            # Brown/tan mask (sand/soil - land indicator)
            brown_mask = cv2.inRange(hsv_image, (10, 30, 80), (30, 200, 220))
            
            # Determine sampling based on orientation
            if is_vertical:
                # For vertical shorelines, focus more on horizontal sampling
                left_sample_points = sample_transect_points(
                    image, middle_point, normal_left, transect_length, is_vertical=True
                )
                right_sample_points = sample_transect_points(
                    image, middle_point, normal_right, transect_length, is_vertical=True
                )
            else:
                # Standard sampling for horizontal shorelines
                left_sample_points = sample_transect_points(
                    image, middle_point, normal_left, transect_length, is_vertical=False
                )
                right_sample_points = sample_transect_points(
                    image, middle_point, normal_right, transect_length, is_vertical=False
                )
            
            # Initialize feature counts
            left_features = {"blue": 0, "dark": 0, "green": 0, "brown": 0}
            right_features = {"blue": 0, "dark": 0, "green": 0, "brown": 0}
            
            # Process left samples
            for y, x in left_sample_points:
                # Count features
                left_features["blue"] += blue_mask[y, x] > 0
                left_features["dark"] += dark_mask[y, x] > 0
                left_features["green"] += green_mask[y, x] > 0
                left_features["brown"] += brown_mask[y, x] > 0
                
                # Mark sample point on debug image
                color = (0, 0, 255) if blue_mask[y, x] > 0 else \
                       (0, 255, 0) if green_mask[y, x] > 0 else \
                       (165, 42, 42) if brown_mask[y, x] > 0 else \
                       (211, 211, 211)  # Default color
                cv2.circle(debug_img, (x, y), 3, color, -1)
                
            # Process right samples
            for y, x in right_sample_points:
                # Count features
                right_features["blue"] += blue_mask[y, x] > 0
                right_features["dark"] += dark_mask[y, x] > 0
                right_features["green"] += green_mask[y, x] > 0
                right_features["brown"] += brown_mask[y, x] > 0
                
                # Mark sample point on debug image
                color = (0, 0, 255) if blue_mask[y, x] > 0 else \
                       (0, 255, 0) if green_mask[y, x] > 0 else \
                       (165, 42, 42) if brown_mask[y, x] > 0 else \
                       (211, 211, 211)  # Default color
                cv2.circle(debug_img, (x, y), 3, color, -1)
                
            # Save debug image directly to model directory
            component_analysis_path = os.path.join(model_dir, "component_analysis_improved.jpg")
            cv2.imwrite(component_analysis_path, cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
            
            # Calculate sea scores with appropriate weighting based on orientation
            if is_vertical:
                # For vertical shorelines, emphasize water indicators more
                left_sea_score = (left_features["blue"] * 3) + (left_features["dark"] * 2) - \
                                 (left_features["green"] * 1.5) - (left_features["brown"] * 1)
                                 
                right_sea_score = (right_features["blue"] * 3) + (right_features["dark"] * 2) - \
                                  (right_features["green"] * 1.5) - (right_features["brown"] * 1)
            else:
                # Standard weights for horizontal shorelines
                left_sea_score = (left_features["blue"] * 2) + (left_features["dark"] * 1) - \
                                 (left_features["green"] * 2) - (left_features["brown"] * 3)
                                 
                right_sea_score = (right_features["blue"] * 2) + (right_features["dark"] * 1) - \
                                  (right_features["green"] * 2) - (right_features["brown"] * 3)
                              
            # The rest of the logic to determine sea direction
            
            # --- TEXTURE ANALYSIS --- (similar to your original function)
            left_variance = 0
            right_variance = 0
            
            # Reduced weight for texture analysis in vertical shorelines
            texture_weight = 0.5 if is_vertical else 1.0
            
            # Convert to grayscale for texture analysis
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Sample points for texture analysis (use fewer points for efficiency)
            left_texture_samples = left_sample_points[::3]  # Every 3rd point
            right_texture_samples = right_sample_points[::3]  # Every 3rd point
            
            # Process samples for variance calculation
            for y, x in left_texture_samples:
                if y >= 5 and y < h-5 and x >= 5 and x < w-5:
                    patch = gray_image[y-5:y+6, x-5:x+6]
                    left_variance += np.var(patch)
                    
            for y, x in right_texture_samples:
                if y >= 5 and y < h-5 and x >= 5 and x < w-5:
                    patch = gray_image[y-5:y+6, x-5:x+6]
                    right_variance += np.var(patch)
                    
            # Normalize by number of samples
            left_variance /= len(left_texture_samples) if left_texture_samples else 1
            right_variance /= len(right_texture_samples) if right_texture_samples else 1
            
            # Initialize vote counters for sea direction
            left_sea_votes = 0
            right_sea_votes = 0
            
            # Method 1: Color-based (highest weight)
            if left_sea_score > right_sea_score:
                left_sea_votes += 3
            else:
                right_sea_votes += 3
                
            # Method 2: Texture (sea has lower variance) with adjusted weight
            if left_variance < right_variance:
                left_sea_votes += texture_weight
            else:
                right_sea_votes += texture_weight
                
            # Add additional vote weight for vertical shorelines based on absolute blue counts
            if is_vertical:
                # Give extra weight to pure blue count
                if left_features["blue"] > right_features["blue"]:
                    left_sea_votes += 2
                else:
                    right_sea_votes += 2
                    
            # Determine final sea direction
            if left_sea_votes >= right_sea_votes:
                sea_direction = normal_left
            else:
                sea_direction = normal_right
                
            # Create final visualization
            final_img = image.copy()
            
            # Draw shoreline
            for point in shoreline_points:
                y, x = int(point[0]), int(point[1])
                if 0 <= y < h and 0 <= x < w:
                    cv2.circle(final_img, (x, y), 1, (255, 255, 255), -1)
                    
            # Draw sea direction arrow
            arrow_start = middle_point
            arrow_end = middle_point + sea_direction * 50
            
            cv2.arrowedLine(final_img,
                            (int(arrow_start[1]), int(arrow_start[0])),
                            (int(arrow_end[1]), int(arrow_end[0])),
                            (0, 255, 255), 3, tipLength=0.3)
                            
            # Add SEA and LAND labels
            sea_label_pos = middle_point + sea_direction * 70
            land_label_pos = middle_point - sea_direction * 70
            
            # Draw text backgrounds to make labels more visible
            cv2.putText(final_img, "SEA",
                       (int(sea_label_pos[1])-2, int(sea_label_pos[0])+2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
            cv2.putText(final_img, "LAND",
                       (int(land_label_pos[1])-2, int(land_label_pos[0])+2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
                       
            # Draw text
            cv2.putText(final_img, "SEA",
                       (int(sea_label_pos[1]), int(sea_label_pos[0])),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(final_img, "LAND",
                       (int(land_label_pos[1]), int(land_label_pos[0])),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                       
            # Save final visualization with orientation indicator directly to model directory
            sea_direction_path = os.path.join(model_dir, "sea_direction_result_improved.jpg")
            cv2.imwrite(sea_direction_path, cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))
            
            return sea_direction
            
        # Fallback default direction
        return np.array([0, 1])
    
    # Override the visualize_baseline_and_transects function with a full implementation
    def model_visualize_baseline_and_transects(image, shoreline_points, baseline_points, transects, sea_direction):
        """A full implementation that saves files to the model directory"""
        import matplotlib.pyplot as plt
        import numpy as np
        import cv2
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Create mask for shoreline
        shoreline_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        for point in shoreline_points:
            y, x = int(point[0]), int(point[1])
            if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
                shoreline_mask[y, x] = 1
        
        # Create mask for baseline with line drawing to ensure continuity
        baseline_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # First, plot points
        for point in baseline_points:
            y, x = int(point[0]), int(point[1])
            if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
                baseline_mask[y, x] = 1
        
        # Second, connect adjacent points with lines to ensure continuity
        for i in range(len(baseline_points)-1):
            pt1 = (int(baseline_points[i][1]), int(baseline_points[i][0]))
            pt2 = (int(baseline_points[i+1][1]), int(baseline_points[i+1][0]))
            cv2.line(baseline_mask, pt1, pt2, 1, 1)  # Draw line between points
        
        # Create color overlay
        overlay = image.copy()
        
        # Add shoreline (blue)
        overlay[shoreline_mask > 0] = [0, 0, 255]  # Blue for shoreline
        
        # Add baseline (red) 
        overlay[baseline_mask > 0] = [255, 0, 0]  # Red for baseline
        
        # Blend with original image for better visibility
        alpha = 0.7
        result = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
        
        # Display the image with colored masks
        plt.imshow(result)
        
        # Plot transects with reduced opacity to avoid cluttering
        max_display_transects = 100
        step = max(1, len(transects) // max_display_transects)
        
        for i in range(0, len(transects), step):
            start_point, end_point = transects[i]
            plt.plot([start_point[1], end_point[1]], [start_point[0], end_point[0]], 
                     'y-', linewidth=1, alpha=0.6)
        
        # Add sea direction indicator at center point
        if len(shoreline_points) > 0:
            center_idx = len(shoreline_points) // 2
            center_point = shoreline_points[center_idx]
            
            # Draw arrow pointing to sea
            plt.arrow(center_point[1], center_point[0],
                      sea_direction[1]*30, sea_direction[0]*30,
                      color='cyan', width=2, head_width=10, head_length=10)
            
            # Ensure label positions are within image boundaries
            h, w = image.shape[:2]
            
            # Add SEA label
            sea_label_point = center_point + sea_direction * 70
            sea_y = int(np.clip(sea_label_point[0], 30, h-30))
            sea_x = int(np.clip(sea_label_point[1], 50, w-50))
            
            plt.text(sea_x, sea_y, "SEA",
                     color='cyan', fontsize=14, fontweight='bold',
                     ha='center', va='center',
                     bbox=dict(facecolor='black', alpha=0.7, edgecolor='none'))
            
            # Add LAND label
            land_label_point = center_point - sea_direction * 70
            land_y = int(np.clip(land_label_point[0], 30, h-30))
            land_x = int(np.clip(land_label_point[1], 50, w-50))
            
            plt.text(land_x, land_y, "LAND",
                     color='green', fontsize=14, fontweight='bold',
                     ha='center', va='center',
                     bbox=dict(facecolor='black', alpha=0.7, edgecolor='none'))
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], color='blue', lw=2, label='Shoreline'),
            plt.Line2D([0], [0], color='red', lw=2, label='Landward Baseline'),
            plt.Line2D([0], [0], color='yellow', lw=1, alpha=0.6, label='Transect')
        ]
        plt.legend(handles=legend_elements, loc='upper right', framealpha=0.8, facecolor='white')
        
        # Add title with background for visibility
        plt.title("Shoreline Analysis with Landward Baseline and Transects", 
                  bbox=dict(facecolor='white', alpha=0.7))
        
        # Remove axes for cleaner visualization
        plt.xticks([])
        plt.yticks([])
        
        # Save figure directly to model directory
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, "baseline_and_transects.png"), dpi=300)
        
        # Create OpenCV version for comparison
        cv_image = image.copy()
        
        # Create separate RGB image with colored masks for precise control
        colored_mask = np.zeros_like(cv_image)
        colored_mask[shoreline_mask > 0] = [255, 0, 0]  # BGR format - Blue for shoreline
        colored_mask[baseline_mask > 0] = [0, 0, 255]  # BGR format - Red for baseline
        
        # Blend with original image
        cv_image = cv2.addWeighted(cv_image, 0.7, colored_mask, 0.8, 0)
        
        # Draw transects
        for i in range(0, len(transects), step):
            start = (int(transects[i][0][1]), int(transects[i][0][0]))
            end = (int(transects[i][1][1]), int(transects[i][1][0]))
            cv2.line(cv_image, start, end, (0, 255, 255), 1)  # Yellow
        
        # Add direction arrow
        if len(shoreline_points) > 0:
            start = (int(center_point[1]), int(center_point[0]))
            end = (int(center_point[1] + sea_direction[1]*30), 
                   int(center_point[0] + sea_direction[0]*30))
            cv2.arrowedLine(cv_image, start, end, (255, 255, 0), 2, tipLength=0.3)
            
            # Add labels
            cv2.putText(cv_image, "SEA", (sea_x, sea_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(cv_image, "SEA", (sea_x, sea_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        
            cv2.putText(cv_image, "LAND", (land_x, land_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(cv_image, "LAND", (land_x, land_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add legend
        cv2.putText(cv_image, "Blue: Shoreline", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(cv_image, "Red: Landward Baseline", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Save OpenCV version directly to model directory
        cv2.imwrite(os.path.join(model_dir, "baseline_and_transects_cv.jpg"), cv_image)
        
        plt.close()
        
        print("Visualizations saved with connected baseline lines")
    
    # Run the analysis
    try:
        # Extract shoreline points
        print("Processing first mask")
        shoreline_points1 = np.argwhere(image1 > 0)
        shoreline1 = order_shoreline_points(shoreline_points1)

        print("Processing second mask")
        shoreline_points2 = np.argwhere(image2 > 0)
        shoreline2 = order_shoreline_points(shoreline_points2)

        print("Detecting sea direction using robust component analysis")
        sea_direction = model_identify_sea_component(satelite1, shoreline1, transect_length=80)

        print(f"Creating landward baseline")
        baseline = create_landward_baseline(
            shoreline1, 
            sea_direction, 
            offset_distance=70,
            image_shape=image1.shape[:2]
        )

        # Ensure baseline has enough points
        if len(baseline) < 5:
            print("WARNING: Created baseline too short. Increasing number of baseline points.")
            baseline = create_landward_baseline(
                shoreline1, 
                sea_direction, 
                offset_distance=70,
                image_shape=image1.shape[:2]
            )
        
        # Generate transects from the landward baseline
        print(f"Generating transects from landward baseline")
        transects = generate_transects_from_baseline(
            baseline, 
            sea_direction,
            num_transects=100,
            transect_length=min(image1.shape[0], image1.shape[1])/3
        )

        # Visualize the baseline and transects
        model_visualize_baseline_and_transects(satelite1, shoreline1, baseline, transects, sea_direction)

        # Calculate NSM and EPR with the new transects
        print("Calculating NSM and EPR with transects from landward baseline")
        nsm_values, epr_values, intersection_points1, intersection_points2, valid_transects = calculate_shoreline_change_with_direction(
            image1, image2, transects,
            pixel_to_meter=pixel_to_meter,
            time_interval_years=(datetime.strptime(date2, '%Y-%m-%d') - datetime.strptime(date1, '%Y-%m-%d')).days / 365.25
        )

        # Modify visualize_shoreline_change to save directly to model directory
        # This function already takes model_name as a parameter, so it can be used as is

        # Visualize results
        print("Generating change visualization")
        stats = visualize_shoreline_change(
            satelite1, satelite2, model_name, shoreline1, shoreline2, valid_transects,
            intersection_points1, intersection_points2, nsm_values
        )

        # Add EPR statistics
        if len(epr_values) > 0:
            stats["avg_epr"] = np.mean(epr_values)
            stats["max_epr"] = np.max(epr_values)
            stats["min_epr"] = np.min(epr_values)
            stats["std_epr"] = np.std(epr_values)
            
            # Save detailed results to CSV in model directory
            results_df = pd.DataFrame({
                'Transect': range(1, len(nsm_values) + 1),
                'NSM (m)': nsm_values,
                'EPR (m/year)': epr_values
            })
            results_df.to_csv(os.path.join(model_dir, "shoreline_results.csv"), index=False)
            
            print(f"Average EPR: {stats['avg_epr']:.2f} meters/year")
            print(f"Results saved to {model_dir}/shoreline_results.csv")
        else:
            print("WARNING: No valid transect intersections were found!")
            stats["avg_epr"] = 0
            stats["max_epr"] = 0
            stats["min_epr"] = 0
            stats["std_epr"] = 0
        
        # Close all matplotlib figures to prevent memory leaks
        plt.close('all')
        
        # Return the EPR and NSM values
        return stats
    
    except Exception as e:
        # Close all matplotlib figures on error too
        plt.close('all')
        print(f"Error in shoreline analysis: {str(e)}")
        raise e

if __name__ == "__main__":
    # Load the sample images
    image1 = "DeepLab_segmented_1_sentinel2_void_2023-10-14_Weligama.jpg"
    image2 = "DeepLab_segmented_2_sentinel2_void_2023-11-18_Weligama.jpg"
    
    # Load satellite images for visualization (assuming they're in the same directory)
    satelite1 = cv2.imread("sentinel2_void_2023-10-14_Weligama.jpg")
    satelite2 = cv2.imread("sentinel2_void_2023-11-18_Weligama.jpg")
    
    # Convert to RGB if loaded images are not None
    if satelite1 is not None:
        satelite1 = cv2.cvtColor(satelite1, cv2.COLOR_BGR2RGB)
    else:
        # Create placeholder if image can't be loaded
        satelite1 = np.zeros((540, 540, 3), dtype=np.uint8)
        
    if satelite2 is not None:
        satelite2 = cv2.cvtColor(satelite2, cv2.COLOR_BGR2RGB)
    else:
        # Create placeholder if image can't be loaded
        satelite2 = np.zeros((540, 540, 3), dtype=np.uint8)

    # Run the analysis with different models
    models = ["U-Net", "DeepLab", "FCN8"]
    
    for model_name in models:
        try:
            # Get proper file paths for this model
            image1_path = f"{model_name}_segmented_1_sentinel2_void_2023-10-14_Weligama.jpg"
            image2_path = f"{model_name}_segmented_2_sentinel2_void_2023-11-18_Weligama.jpg"
            
            # Run analysis and get stats
            stats = run_shoreline_analysis(image1_path, image2_path, satelite1, satelite2, model_name)
            
            # Access specific stats from the returned dictionary
            print(f"\n{model_name} Results Summary:")
            print(f"Average EPR: {stats['avg_epr']:.2f} m/year")
            print(f"Average NSM: {stats['avg_nsm']:.2f} m")
            print(f"Max Erosion: {abs(stats['min_nsm']):.2f} m")
            print(f"Max Accretion: {stats['max_nsm']:.2f} m")
            print(f"Erosion percentage: {stats['erosion_percent']:.1f}%")
            print(f"Accretion percentage: {stats['accretion_percent']:.1f}%")
            print(f"All results saved to analysis_results/{model_name}/")
            
        except Exception as e:
            print(f"Error running analysis for {model_name}: {str(e)}")
