import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import torch

def sample_transect_points(image, start_point, direction, length, is_vertical=False):
    h, w = image.shape[:2]
    points = []
    num_samples = 20  # Use a consistent sample number instead of conditional
    
    for i in range(1, num_samples + 1):
        step = (i / num_samples) * length
        sample_point = np.round(start_point + direction * step).astype(int)
        if 0 <= sample_point[0] < h and 0 <= sample_point[1] < w:
            points.append((sample_point[0], sample_point[1]))
    return points


def identify_sea_component(image, shoreline_points, model_dir=None, transect_length=80):
    if len(shoreline_points) == 0:
        return np.array([0, 1])

    h, w = image.shape[:2]

    if len(shoreline_points) >= 10:
        first_point = shoreline_points[0]
        last_point = shoreline_points[-1]

        shoreline_direction = last_point - first_point
        shoreline_length = np.linalg.norm(shoreline_direction)

        if shoreline_length < 10:
            return np.array([0, 1])

        shoreline_direction = shoreline_direction / shoreline_length

        normal_left = np.array([-shoreline_direction[1], shoreline_direction[0]])
        normal_right = -normal_left

        middle_idx = len(shoreline_points) // 2
        middle_point = shoreline_points[middle_idx]

        debug_img = image.copy()

        for point in shoreline_points:
            y, x = int(point[0]), int(point[1])
            if 0 <= y < h and 0 <= x < w:
                cv2.circle(debug_img, (x, y), 1, (255, 255, 255), -1)

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

        cv2.putText(debug_img, "Component 1",
                   (int(middle_point[1] + normal_left[1]*arrow_length*1.2),
                    int(middle_point[0] + normal_left[0]*arrow_length*1.2)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(debug_img, "Component 2",
                   (int(middle_point[1] + normal_right[1]*arrow_length*1.2),
                    int(middle_point[0] + normal_right[0]*arrow_length*1.2)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        blue_mask = cv2.inRange(hsv_image, (90, 20, 30), (150, 255, 255))
        dark_mask = cv2.inRange(hsv_image, (0, 0, 0), (180, 50, 100))
        green_mask = cv2.inRange(hsv_image, (35, 30, 30), (85, 255, 255))
        brown_mask = cv2.inRange(hsv_image, (10, 30, 80), (30, 200, 220))

        left_sample_points = sample_transect_points(
            image, middle_point, normal_left, transect_length, is_vertical=False
        )
        right_sample_points = sample_transect_points(
            image, middle_point, normal_right, transect_length, is_vertical=False
        )

        left_features = {"blue": 0, "dark": 0, "green": 0, "brown": 0}
        right_features = {"blue": 0, "dark": 0, "green": 0, "brown": 0}

        for y, x in left_sample_points:
            left_features["blue"] += blue_mask[y, x] > 0
            left_features["dark"] += dark_mask[y, x] > 0
            left_features["green"] += green_mask[y, x] > 0
            left_features["brown"] += brown_mask[y, x] > 0

            color = (0, 0, 255) if blue_mask[y, x] > 0 else \
                   (0, 255, 0) if green_mask[y, x] > 0 else \
                   (165, 42, 42) if brown_mask[y, x] > 0 else \
                   (211, 211, 211)
            cv2.circle(debug_img, (x, y), 3, color, -1)

        for y, x in right_sample_points:
            right_features["blue"] += blue_mask[y, x] > 0
            right_features["dark"] += dark_mask[y, x] > 0
            right_features["green"] += green_mask[y, x] > 0
            right_features["brown"] += brown_mask[y, x] > 0

            color = (0, 0, 255) if blue_mask[y, x] > 0 else \
                   (0, 255, 0) if green_mask[y, x] > 0 else \
                   (165, 42, 42) if brown_mask[y, x] > 0 else \
                   (211, 211, 211)
            cv2.circle(debug_img, (x, y), 3, color, -1)

        left_sea_score = (left_features["blue"] * 2) + (left_features["dark"] * 1) - \
                        (left_features["green"] * 2) - (left_features["brown"] * 10)

        right_sea_score = (right_features["blue"] * 2) + (right_features["dark"] * 1) - \
                          (right_features["green"] * 2) - (right_features["brown"] * 10)

        print("--- Component Analysis ---")
        print(f"Component 1 (Left) - Blue: {left_features['blue']}, Dark: {left_features['dark']}, " +
              f"Green: {left_features['green']}, Brown: {left_features['brown']}")
        print(f"Component 2 (Right) - Blue: {right_features['blue']}, Dark: {right_features['dark']}, " +
              f"Green: {right_features['green']}, Brown: {right_features['brown']}")
        print(f"Component 1 Sea Score: {left_sea_score}")
        print(f"Component 2 Sea Score: {right_sea_score}")

        left_variance = 0
        right_variance = 0
        texture_weight = 1.0
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        left_texture_samples = left_sample_points[::3]
        right_texture_samples = right_sample_points[::3]

        for y, x in left_texture_samples:
            if y >= 5 and y < h-5 and x >= 5 and x < w-5:
                patch = gray_image[y-5:y+6, x-5:x+6]
                left_variance += np.var(patch)

        for y, x in right_texture_samples:
            if y >= 5 and y < h-5 and x >= 5 and x < w-5:
                patch = gray_image[y-5:y+6, x-5:x+6]
                right_variance += np.var(patch)

        left_variance /= len(left_texture_samples) if left_texture_samples else 1
        right_variance /= len(right_texture_samples) if right_texture_samples else 1

        left_sea_votes = 0
        right_sea_votes = 0

        if left_sea_score > right_sea_score:
            left_sea_votes += 3
        else:
            right_sea_votes += 3

        if left_variance < right_variance:
            left_sea_votes += texture_weight
        else:
            right_sea_votes += texture_weight

        if left_features["blue"] > right_features["blue"]:
            left_sea_votes += 2
        else:
            right_sea_votes += 2

        if left_sea_votes >= right_sea_votes:
            print("DECISION: Component 1 (LEFT) is SEA")
            sea_direction = normal_left
        else:
            print("DECISION: Component 2 (RIGHT) is SEA")
            sea_direction = normal_right

        final_img = image.copy()

        for point in shoreline_points:
            y, x = int(point[0]), int(point[1])
            if 0 <= y < h and 0 <= x < w:
                cv2.circle(final_img, (x, y), 1, (255, 255, 255), -1)

        arrow_start = middle_point
        arrow_end = middle_point + sea_direction * 50

        cv2.arrowedLine(final_img,
                        (int(arrow_start[1]), int(arrow_start[0])),
                        (int(arrow_end[1]), int(arrow_end[0])),
                        (0, 255, 255), 3, tipLength=0.3)

        sea_label_pos = middle_point + sea_direction * 70
        land_label_pos = middle_point - sea_direction * 70

        cv2.putText(final_img, "SEA",
                   (int(sea_label_pos[1])-2, int(sea_label_pos[0])+2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
        cv2.putText(final_img, "LAND",
                   (int(land_label_pos[1])-2, int(land_label_pos[0])+2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)

        cv2.putText(final_img, "SEA",
                   (int(sea_label_pos[1]), int(sea_label_pos[0])),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(final_img, "LAND",
                   (int(land_label_pos[1]), int(land_label_pos[0])),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Save to model directory if provided, otherwise use current directory
        sea_direction_path = "sea_direction_result_improved.jpg"
        if model_dir:
            sea_direction_path = os.path.join(model_dir, "sea_direction_result_improved.jpg")
        cv2.imwrite(sea_direction_path, cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))

        return sea_direction

    return np.array([0, 1])

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

def order_shoreline_points(points, max_gap=10000):
    if len(points) == 0:
        return np.array([])

    ordered = [points[np.argmin(points[:, 1])]]
    remaining = list(range(len(points)))
    remaining.remove(np.argmin(points[:, 1]))

    while remaining:
        last_point = ordered[-1]
        distances = np.sqrt(np.sum((points[remaining] - last_point)**2, axis=1))
        closest_idx = np.argmin(distances)
        closest_point_idx = remaining[closest_idx]

        if distances[closest_idx] > max_gap:
            if len(remaining) > len(points) / 2:
                leftmost_idx = np.argmin(points[remaining][:, 1])
                ordered.append(points[remaining[leftmost_idx]])
                remaining.pop(leftmost_idx)
            else:
                break
        else:
            ordered.append(points[closest_point_idx])
            remaining.pop(closest_idx)

    return np.array(ordered)

def calculate_actual_shoreline_distance(shoreline_points):
    """
    Calculate cumulative actual distance along the shoreline points.
    """
    if len(shoreline_points) < 2:
        return np.array([0])

    # Calculate distances between consecutive points
    distances = np.zeros(len(shoreline_points))
    for i in range(1, len(shoreline_points)):
        # Euclidean distance between consecutive points
        distances[i] = np.linalg.norm(shoreline_points[i] - shoreline_points[i-1])

    # Calculate cumulative distance
    cumulative_distances = np.cumsum(distances)

    return cumulative_distances

def create_landward_baseline(shoreline, sea_direction, offset_distance=70, image_shape=None):
    """Create a landward baseline parallel to the shoreline"""
    if len(shoreline) == 0:
        return []
    
    # Move shoreline points in direction away from sea
    baseline = shoreline - sea_direction * offset_distance
    
    # Clip to image boundaries if provided
    if image_shape is not None:
        h, w = image_shape
        baseline[:, 0] = np.clip(baseline[:, 0], 0, h-1)
        baseline[:, 1] = np.clip(baseline[:, 1], 0, w-1)
    
    return baseline

def generate_transects_from_baseline(baseline, sea_direction, num_transects=100, transect_length=200):
    """Generate transects from baseline points toward the sea"""
    if len(baseline) < 2:
        return []
    
    # Select evenly spaced points along the baseline
    indices = np.linspace(0, len(baseline)-1, num_transects).astype(int)
    selected_points = baseline[indices]
    
    # Create transects: start from baseline point, extend in sea direction
    transects = []
    for point in selected_points:
        # Start at baseline point
        start_point = point
        # End point is in the sea direction
        end_point = point + sea_direction * transect_length
        transects.append((start_point, end_point))
    
    return transects

# def generate_equally_spaced_transects(shoreline, spacing_meters=100, transect_length=100,
#                                      sea_direction=None, pixel_to_meter=18.2):
#     """
#     Generate transects with more precise equal spacing along the shoreline.
#     """
#     if len(shoreline) < 2:
#         print("Warning: Not enough shoreline points to generate transects")
#         return []

#     # Calculate the exact cumulative distances between all points
#     cumulative_distances = np.zeros(len(shoreline))
#     for i in range(1, len(shoreline)):
#         # Calculate Euclidean distance between consecutive points
#         distance = np.linalg.norm(shoreline[i] - shoreline[i-1])
#         cumulative_distances[i] = cumulative_distances[i-1] + distance

#     # Convert spacing from meters to pixels
#     spacing_pixels = spacing_meters / pixel_to_meter

#     # Calculate total shoreline length
#     total_length = cumulative_distances[-1]
#     total_length_meters = total_length * pixel_to_meter

#     # Calculate exact positions for transects at equal intervals
#     num_transects = max(2, int(total_length / spacing_pixels))
#     exact_distances = np.linspace(0, total_length, num_transects)

#     # Generate transects at precisely these positions
#     transects = []
#     for target_distance in exact_distances:
#         # Binary search to find the exact position along the shoreline
#         left = 0
#         right = len(cumulative_distances) - 1

#         while right - left > 1:
#             mid = (left + right) // 2
#             if cumulative_distances[mid] < target_distance:
#                 left = mid
#             else:
#                 right = mid

#         # Calculate interpolation factor between the two closest points
#         segment_length = cumulative_distances[right] - cumulative_distances[left]
#         if segment_length > 0:
#             alpha = (target_distance - cumulative_distances[left]) / segment_length
#         else:
#             alpha = 0

#         # Interpolate to get the exact point
#         point = shoreline[left] * (1 - alpha) + shoreline[right] * alpha

#         # Calculate local orientation using nearby points
#         window = 3  # Use nearby points for better direction estimation
#         start_idx = max(0, left - window)
#         end_idx = min(len(shoreline) - 1, right + window)

#         if end_idx > start_idx + 1:
#             direction_points = shoreline[start_idx:end_idx+1]
#             # Use linear regression to find the best direction
#             y_coords = direction_points[:, 0]
#             x_coords = direction_points[:, 1]

#             A = np.vstack([x_coords, np.ones(len(x_coords))]).T
#             slope, _ = np.linalg.lstsq(A, y_coords, rcond=None)[0]

#             tangent = np.array([slope, 1])
#             tangent = tangent / np.linalg.norm(tangent)
#         else:
#             # Fallback to direct direction if not enough points
#             if right < len(shoreline) - 1:
#                 tangent = shoreline[right+1] - shoreline[right]
#             else:
#                 tangent = shoreline[right] - shoreline[left]
#             tangent = tangent / np.linalg.norm(tangent)

#         # Calculate normal vector (perpendicular to tangent)
#         normal = np.array([-tangent[1], tangent[0]])

#         # Orient toward sea if direction provided
#         if sea_direction is not None:
#             if np.dot(normal, sea_direction) < 0:
#                 normal = -normal

#         # Generate transect start and end points
#         start_point = point - normal * transect_length/2
#         end_point = point + normal * transect_length/2
#         transects.append((start_point, end_point))

#     return transects
def generate_truly_perpendicular_transects(shoreline, spacing_meters=100, transect_length=100,
                                  sea_direction=None, pixel_to_meter=1.0, debug_output=True):
    """
    Generate transects that are truly perpendicular to the shoreline with extensive validation.
    
    Parameters:
    -----------
    shoreline : numpy.ndarray
        Array of shoreline points in format [row, col]
    spacing_meters : float
        Desired spacing between transects in meters
    transect_length : float
        Length of transects in pixels
    sea_direction : numpy.ndarray or None
        Direction vector pointing towards the sea
    pixel_to_meter : float
        Conversion factor from pixels to meters
    debug_output : bool
        Whether to print debug information and save debug visualizations
        
    Returns:
    --------
    dict
        Dictionary containing transects and debug information
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2
    import os
    from scipy.signal import savgol_filter
    
    if len(shoreline) < 10:
        print("Warning: Not enough shoreline points to generate transects")
        return {"transects": [], "debug_info": None}
    
    results = {
        "transects": [],
        "debug_info": {
            "shoreline_points": shoreline,
            "tangent_vectors": [],
            "normal_vectors": [],
            "transect_points": [],
            "angles": []
        }
    }

    print(f"Starting with {len(shoreline)} shoreline points")
    
    # Step 1: Smooth the shoreline to reduce noise
    # First convert to separate x and y arrays
    shoreline_rows = shoreline[:, 0]
    shoreline_cols = shoreline[:, 1]
    
    # Apply Savitzky-Golay filter to smooth the shoreline
    window_length = min(51, len(shoreline) - (len(shoreline) % 2 + 1))  # Must be odd and smaller than array
    if window_length < 5:
        # Not enough points for smoothing
        smoothed_rows = shoreline_rows
        smoothed_cols = shoreline_cols
    else:
        try:
            poly_order = min(3, window_length - 1)
            smoothed_rows = savgol_filter(shoreline_rows, window_length, poly_order)
            smoothed_cols = savgol_filter(shoreline_cols, window_length, poly_order)
        except Exception as e:
            print(f"Smoothing error: {str(e)}. Using original shoreline.")
            smoothed_rows = shoreline_rows
            smoothed_cols = shoreline_cols
    
    # Recombine into a single array
    smoothed_shoreline = np.column_stack((smoothed_rows, smoothed_cols))
    
    if debug_output:
        # Visualize original vs smoothed shoreline
        plt.figure(figsize=(12, 8))
        plt.plot(shoreline[:, 1], shoreline[:, 0], 'b-', alpha=0.5, label='Original')
        plt.plot(smoothed_shoreline[:, 1], smoothed_shoreline[:, 0], 'r-', label='Smoothed')
        plt.legend()
        plt.title('Original vs Smoothed Shoreline')
        plt.axis('equal')
        plt.grid(True)
        plt.savefig(os.path.join("analysis_results", "shoreline_smoothing.png"), dpi=300)
        plt.close()
    
    # Step 2: Calculate the cumulative distance along the shoreline
    cumulative_distances = np.zeros(len(smoothed_shoreline))
    for i in range(1, len(smoothed_shoreline)):
        distance = np.linalg.norm(smoothed_shoreline[i] - smoothed_shoreline[i-1])
        cumulative_distances[i] = cumulative_distances[i-1] + distance
    
    total_length = cumulative_distances[-1]
    print(f"Total shoreline length: {total_length:.2f} pixels ({total_length*pixel_to_meter:.2f} meters)")
    
    # Convert spacing from meters to pixels
    spacing_pixels = spacing_meters / pixel_to_meter
    
    # Calculate the number of transects
    num_transects = max(3, int(total_length / spacing_pixels))
    print(f"Generating {num_transects} transects with {spacing_pixels:.2f} pixel spacing")
    
    # Calculate exact positions for transects at equal intervals
    target_distances = np.linspace(0, total_length, num_transects)
    
    # Create visualization image for debugging
    if debug_output:
        debug_img = np.zeros((1500, 1500, 3), dtype=np.uint8)
        # Draw smoothed shoreline
        for i in range(1, len(smoothed_shoreline)):
            pt1 = (int(smoothed_shoreline[i-1][1]), int(smoothed_shoreline[i-1][0]))
            pt2 = (int(smoothed_shoreline[i][1]), int(smoothed_shoreline[i][0]))
            cv2.line(debug_img, pt1, pt2, (255, 255, 255), 1)
    
    # Step 3: Generate transects at each position
    for idx, target_distance in enumerate(target_distances):
        # Find the segment containing this distance
        segment_idx = np.searchsorted(cumulative_distances, target_distance) - 1
        if segment_idx < 0:
            segment_idx = 0
        
        # Get the segment endpoints
        if segment_idx >= len(smoothed_shoreline) - 1:
            segment_idx = len(smoothed_shoreline) - 2
        
        p1 = smoothed_shoreline[segment_idx]
        p2 = smoothed_shoreline[segment_idx + 1]
        
        # Calculate the interpolation factor
        segment_length = cumulative_distances[segment_idx + 1] - cumulative_distances[segment_idx]
        if segment_length > 0:
            alpha = (target_distance - cumulative_distances[segment_idx]) / segment_length
        else:
            alpha = 0
        
        # Interpolate to get the exact point on the shoreline
        point = p1 * (1 - alpha) + p2 * alpha
        
        # Calculate the tangent directly using segment direction
        # Note: We're calculating tangent using direct segment direction, not linear regression
        # This ensures a more accurate local estimate
        window = 5  # Number of points to consider on each side
        start_idx = max(0, segment_idx - window)
        end_idx = min(len(smoothed_shoreline) - 1, segment_idx + window + 1)
        
        # Calculate tangent as weighted average of nearby segments
        tangent = np.zeros(2)
        total_weight = 0
        
        for i in range(start_idx, end_idx - 1):
            seg_p1 = smoothed_shoreline[i]
            seg_p2 = smoothed_shoreline[i + 1]
            seg_vector = seg_p2 - seg_p1
            
            # Weight based on distance from our point
            weight = 1.0 / (1.0 + abs(i - segment_idx))
            seg_length = np.linalg.norm(seg_vector)
            
            if seg_length > 0:
                # Add weighted contribution
                tangent += (seg_vector / seg_length) * weight
                total_weight += weight
        
        if total_weight > 0:
            tangent /= total_weight
            tangent_length = np.linalg.norm(tangent)
            if tangent_length > 0:
                tangent /= tangent_length
        else:
            # Fallback: use the immediate segment if weighted average failed
            segment_vector = p2 - p1
            segment_length = np.linalg.norm(segment_vector)
            if segment_length > 0:
                tangent = segment_vector / segment_length
            else:
                # Last resort: use a horizontal vector
                tangent = np.array([0, 1])
        
        # Calculate the normal (perpendicular) vector
        # In [row, col] format: normal = [-tangent[1], tangent[0]]
        normal = np.array([-tangent[1], tangent[0]])
        
        # Make sure normal vector points toward the sea if sea_direction is provided
        if sea_direction is not None:
            if np.dot(normal, sea_direction) < 0:
                normal = -normal
                
        # Create transect start and end points
        start_point = point - normal * (transect_length / 2)
        end_point = point + normal * (transect_length / 2)
        
        # Calculate angle between tangent and normal (should be 90 degrees)
        dot_product = np.dot(tangent, normal)
        angle_rad = np.arccos(min(1.0, max(-1.0, abs(dot_product))))
        angle_deg = np.degrees(angle_rad)
        
        # Store transect and debug info
        results["transects"].append((start_point, end_point))
        results["debug_info"]["tangent_vectors"].append((point, tangent))
        results["debug_info"]["normal_vectors"].append((point, normal))
        results["debug_info"]["transect_points"].append(point)
        results["debug_info"]["angles"].append(angle_deg)
        
        if debug_output:
            # Draw the transect point
            pt = (int(point[1]), int(point[0]))
            cv2.circle(debug_img, pt, 3, (0, 255, 255), -1)
            
            # Draw tangent vector (magnified for visibility)
            tangent_scale = 30
            tangent_end = (
                int(point[1] + tangent[1] * tangent_scale),
                int(point[0] + tangent[0] * tangent_scale)
            )
            cv2.line(debug_img, pt, tangent_end, (0, 0, 255), 1)
            
            # Draw normal vector (transect direction)
            normal_scale = 50
            normal_end = (
                int(point[1] + normal[1] * normal_scale),
                int(point[0] + normal[0] * normal_scale)
            )
            cv2.line(debug_img, pt, normal_end, (0, 255, 0), 2)
            
            # Draw transect
            start_pt = (int(start_point[1]), int(start_point[0]))
            end_pt = (int(end_point[1]), int(end_point[0]))
            cv2.line(debug_img, start_pt, end_pt, (255, 0, 255), 1)
            
            # Add angle annotation
            if idx % 5 == 0:
                angle_text = f"{angle_deg:.1f}°"
                cv2.putText(debug_img, angle_text, (pt[0] + 10, pt[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    if debug_output:
        # Add legend
        cv2.putText(debug_img, "Yellow: Shoreline Points", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(debug_img, "Red: Tangent", (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(debug_img, "Green: Normal (Transect)", (20, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(debug_img, "Pink: Full Transect", (20, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        cv2.imwrite(os.path.join("analysis_results", "transect_debug.png"), debug_img)
        
        # Calculate statistics
        angles = np.array(results["debug_info"]["angles"])
        perpendicular_count = np.sum(abs(angles - 90) < 5)
        
        print(f"Transect perpendicularity statistics:")
        print(f"  Average deviation from 90°: {np.mean(abs(angles - 90)):.2f}°")
        print(f"  Maximum deviation from 90°: {np.max(abs(angles - 90)):.2f}°")
        print(f"  Minimum angle: {np.min(angles):.2f}°")
        print(f"  Maximum angle: {np.max(angles):.2f}°")
        print(f"  Transects within 5° of perpendicular: {perpendicular_count}/{len(angles)} ({100*perpendicular_count/len(angles):.1f}%)")
        
        # Plot angle histogram
        plt.figure(figsize=(10, 6))
        plt.hist(angles, bins=20, color='blue', alpha=0.7)
        plt.axvline(x=90, color='red', linestyle='--', label='Perfect Perpendicular (90°)')
        plt.xlabel('Angle (degrees)')
        plt.ylabel('Count')
        plt.title('Distribution of Angles between Tangent and Normal')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join("analysis_results", "transect_angles_histogram.png"), dpi=300)
        plt.close()
    
    return results
def verify_true_perpendicularity(shoreline, transects):
    """
    Verify that transects are truly perpendicular to the shoreline.
    This version uses a more direct verification approach by comparing
    the transect direction to the direction of the shoreline segment.
    
    Parameters:
    -----------
    shoreline : numpy.ndarray
        Array of shoreline points
    transects : list
        List of transects, each defined by [start_point, end_point]
        
    Returns:
    --------
    dict
        Dictionary with verification results
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    
    results = {
        "perpendicular_count": 0,
        "non_perpendicular_count": 0,
        "angles": [],
        "transect_details": []
    }
    
    if len(shoreline) < 2 or len(transects) == 0:
        return results
    
    # Visualize the verification
    plt.figure(figsize=(12, 8))
    plt.plot(shoreline[:, 1], shoreline[:, 0], 'k-', linewidth=1, label='Shoreline')
    
    for i, (start, end) in enumerate(transects):
        # Transect direction
        transect_dir = end - start
        transect_length = np.linalg.norm(transect_dir)
        
        if transect_length > 0:
            transect_dir = transect_dir / transect_length
        else:
            continue  # Skip zero-length transects
        
        # Midpoint of transect (should be on shoreline)
        midpoint = (start + end) / 2
        
        # Find nearest point on shoreline
        distances = np.array([np.linalg.norm(p - midpoint) for p in shoreline])
        nearest_idx = np.argmin(distances)
        
        # Estimate local shoreline direction using nearest points
        window = 5
        left_idx = max(0, nearest_idx - window)
        right_idx = min(len(shoreline) - 1, nearest_idx + window)
        
        if right_idx > left_idx + 1:
            # Use linear regression for better direction estimation
            x = shoreline[left_idx:right_idx+1, 1]  # column coordinates
            y = shoreline[left_idx:right_idx+1, 0]  # row coordinates
            
            if len(np.unique(x)) > 1:  # Ensure we have enough unique points
                # Fit line to local shoreline segment
                slope, intercept = np.polyfit(x, y, 1)
                
                # Calculate shoreline direction in [row, col] format
                # If y = mx + b, then direction is [1, 1/m]
                if abs(slope) > 1e-10:  # Avoid division by zero
                    shore_dir = np.array([1, 1/slope])
                    shore_dir = shore_dir / np.linalg.norm(shore_dir)
                else:
                    # Near-vertical shoreline
                    shore_dir = np.array([1, 0])
            else:
                # Vertical line case - use points directly
                shore_dir = shoreline[right_idx] - shoreline[left_idx]
                if np.linalg.norm(shore_dir) > 0:
                    shore_dir = shore_dir / np.linalg.norm(shore_dir)
                else:
                    continue  # Skip problematic segments
        else:
            # Not enough points for regression
            continue
        
        # Calculate angle between transect and shoreline
        # For perpendicular lines, dot product should be close to 0
        dot_product = np.abs(np.dot(transect_dir, shore_dir))
        
        # Ensure we get the angle to perpendicular line (90 degrees)
        angle_rad = np.arccos(min(1.0, max(-1.0, dot_product)))
        angle_deg = np.degrees(angle_rad)
        
        # Store the angle
        results["angles"].append(angle_deg)
        results["transect_details"].append({
            "transect_idx": i,
            "transect_dir": transect_dir,
            "shore_dir": shore_dir,
            "angle": angle_deg,
            "start": start,
            "end": end,
            "midpoint": midpoint,
            "nearest_shore_idx": nearest_idx
        })
        
        # Determine if perpendicular (within 5 degrees)
        if abs(angle_deg - 90) <= 5:
            results["perpendicular_count"] += 1
            # Plot perpendicular transects in green
            plt.plot([start[1], end[1]], [start[0], end[0]], 'g-', alpha=0.7)
        else:
            results["non_perpendicular_count"] += 1
            # Plot non-perpendicular transects in red
            plt.plot([start[1], end[1]], [start[0], end[0]], 'r-', alpha=0.5)
            
        # Add angle annotation for some transects
        if i % 10 == 0:
            plt.text(midpoint[1], midpoint[0], f"{angle_deg:.1f}°", 
                     fontsize=8, color='blue', ha='center', va='bottom')
    
    # Add legend and title
    plt.legend(['Shoreline', 'Perpendicular Transects', 'Non-Perpendicular Transects'])
    plt.title('Verification of Transect Perpendicularity')
    plt.axis('equal')
    plt.grid(True)
    
    # Save the figure
    plt.savefig(os.path.join("analysis_results", "transect_verification.png"), dpi=300)
    plt.close()
    
    # Calculate summary statistics
    if results["angles"]:
        angles = np.array(results["angles"])
        results["average_deviation"] = np.mean(abs(angles - 90))
        results["max_deviation"] = np.max(abs(angles - 90))
        results["min_angle"] = np.min(angles)
        results["max_angle"] = np.max(angles)
        results["percent_perpendicular"] = (results["perpendicular_count"] / len(transects)) * 100
        
        # Plot angle histogram
        plt.figure(figsize=(10, 6))
        plt.hist(angles, bins=20, color='blue', alpha=0.7)
        plt.axvline(x=90, color='red', linestyle='--', label='Perfect Perpendicular (90°)')
        plt.xlabel('Angle (degrees)')
        plt.ylabel('Count')
        plt.title('Distribution of Angles between Transects and Shoreline')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join("analysis_results", "verification_angles_histogram.png"), dpi=300)
        plt.close()
    
    return results

def visualize_transect_spacing(shoreline, transects, model_dir=None):
    plt.figure(figsize=(12, 6))

    # Plot shoreline
    plt.plot(shoreline[:, 1], shoreline[:, 0], 'k-', linewidth=1)

    # Plot transects with numbered labels
    transect_points = []
    for i, (start, end) in enumerate(transects):
        plt.plot([start[1], end[1]], [start[0], end[0]], 'r-', alpha=0.5)
        # Get the point where transect intersects shoreline (midpoint)
        midpoint = (start + end) / 2
        transect_points.append(midpoint)
        # Add transect number
        if i % 5 == 0:  # Label every 5th transect
            plt.text(midpoint[1], midpoint[0], str(i), fontsize=8, ha='center')

    transect_points = np.array(transect_points)

    # Plot distances between consecutive transects
    for i in range(1, len(transect_points)):
        dist = np.linalg.norm(transect_points[i] - transect_points[i-1])
        mid_x = (transect_points[i][1] + transect_points[i-1][1]) / 2
        mid_y = (transect_points[i][0] + transect_points[i-1][0]) / 2
        plt.text(mid_x, mid_y, f"{dist:.1f}", fontsize=6, color='blue')

    plt.title("Transect Spacing Visualization")
    plt.axis('equal')
    plt.grid(True)
    
    # Save to model directory if provided, otherwise use current directory
    output_path = "transect_spacing.png"
    if model_dir:
        output_path = os.path.join(model_dir, "transect_spacing.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

def find_intersection_points_improved(line_start, line_end, mask):
    """
    Find all intersection points where a transect meets a shoreline.
    Returns all intersection points instead of just the median.
    """
    line_start_cv = (int(line_start[1]), int(line_start[0]))
    line_end_cv = (int(line_end[1]), int(line_end[0]))

    blank = np.zeros_like(mask)
    cv2.line(blank, line_start_cv, line_end_cv, 1, 1)
    intersection = np.logical_and(blank, mask).astype(np.uint8)
    points = np.argwhere(intersection > 0)

    if len(points) == 0:
        return []

    # Return all intersection points
    return [np.array([point[0], point[1]]) for point in points]

def get_transect_midpoint(transect):
    """Calculate the midpoint of a transect line."""
    start_point, end_point = transect
    return (start_point + end_point) / 2

def find_closest_point_to_midpoint(points, midpoint):
    """Find the point closest to the transect midpoint."""
    if not points:
        return None

    distances = [np.linalg.norm(point - midpoint) for point in points]
    closest_idx = np.argmin(distances)
    return points[closest_idx]

def calculate_shoreline_change_with_direction(image1, image2, transects, 
                                             pixel_to_meter=1.0, time_interval_years=1.0):
    """Calculate shoreline change along transects"""
    # Create binary masks
    shoreline1_mask = (image1 > 0).astype(np.uint8)
    shoreline2_mask = (image2 > 0).astype(np.uint8)
    
    nsm_values = []
    epr_values = []
    intersection_points1 = []
    intersection_points2 = []
    valid_transects = []
    
    for i, transect in enumerate(transects):
        # Find intersections with both shorelines
        intersections1 = find_intersection_points_improved(transect[0], transect[1], shoreline1_mask)
        intersections2 = find_intersection_points_improved(transect[0], transect[1], shoreline2_mask)
        
        if not intersections1 or not intersections2:
            continue
        
        # Get midpoint of transect
        midpoint = get_transect_midpoint(transect)
        
        # Get closest points to midpoint
        closest_point1 = find_closest_point_to_midpoint(intersections1, midpoint)
        closest_point2 = find_closest_point_to_midpoint(intersections2, midpoint)
        
        if closest_point1 is None or closest_point2 is None:
            continue
        
        valid_transects.append(transect)
        intersection_points1.append(closest_point1)
        intersection_points2.append(closest_point2)
        
        # Calculate distance and direction
        distance = np.linalg.norm(closest_point2 - closest_point1) * pixel_to_meter
        
        # Calculate direction
        movement_vector = closest_point2 - closest_point1
        transect_vector = transect[1] - transect[0]
        transect_vector = transect_vector / np.linalg.norm(transect_vector)
        
        # Determine sign (positive = accretion, negative = erosion)
        # Based on dot product of movement vector and transect vector
        direction = np.dot(movement_vector, transect_vector)
        sign = -1 if direction > 0 else 1
        
        signed_distance = sign * distance
        nsm_values.append(signed_distance)
        
        # Calculate End Point Rate (EPR)
        epr = signed_distance / time_interval_years
        epr_values.append(epr)
    
    return np.array(nsm_values), np.array(epr_values), intersection_points1, intersection_points2, valid_transects

def visualize_shoreline_change(image1, image2, model_name, shoreline1, shoreline2, transects,
                              intersection_points1, intersection_points2, nsm_values):
    """Visualize shoreline change results"""
    model_dir = os.path.join("analysis_results", model_name)
    
    plt.figure(figsize=(12, 10))
    
    # Create a black background image instead of using the input image
    combined_img = np.zeros((image1.shape[0], image1.shape[1], 3), dtype=np.uint8)
    mask1 = np.zeros((image1.shape[0], image1.shape[1]), dtype=np.uint8)
    mask2 = np.zeros((image2.shape[0], image2.shape[1]), dtype=np.uint8)
    
    for point in shoreline1:
        y, x = int(point[0]), int(point[1])
        if 0 <= y < mask1.shape[0] and 0 <= x < mask1.shape[1]:
            mask1[y, x] = 255
    
    for point in shoreline2:
        y, x = int(point[0]), int(point[1])
        if 0 <= y < mask2.shape[0] and 0 <= x < mask2.shape[1]:
            mask2[y, x] = 255
    
    # Create RGB visualization
    combined_mask = np.zeros((image1.shape[0], image1.shape[1], 3), dtype=np.uint8)
    combined_mask[..., 0] = mask1  # Red channel for first shoreline
    combined_mask[..., 2] = mask2  # Blue channel for second shoreline
    
    # No need to blend with background since we're using a black background
    combined_img = combined_mask.copy()
    
    # Draw transects and change indicators
    max_accretion = max(max(nsm_values), 0) if len(nsm_values) > 0 else 0
    max_erosion = abs(min(min(nsm_values), 0)) if len(nsm_values) > 0 else 0
    
    for i, (transect, p1, p2, nsm) in enumerate(zip(transects, intersection_points1, intersection_points2, nsm_values)):
        # Draw transect line
        start = (int(transect[0][1]), int(transect[0][0]))
        end = (int(transect[1][1]), int(transect[1][0]))
        cv2.line(combined_img, start, end, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Draw intersection points
        cv2.circle(combined_img, (int(p1[1]), int(p1[0])), 3, (0, 255, 255), -1)
        cv2.circle(combined_img, (int(p2[1]), int(p2[0])), 3, (255, 255, 0), -1)
        
        # Calculate line width based on magnitude of change
        abs_nsm = abs(nsm)
        max_val = max(max_accretion, max_erosion)
        width = 1 + 4 * (abs_nsm / max_val) if max_val > 0 else 1
        width = min(int(width), 10)  # Cap width
        
        # Draw change line with color based on direction
        if nsm > 0:  # Accretion
            color = (0, 255, 0)  # Green
        else:  # Erosion
            color = (0, 0, 255)  # Red
        
        cv2.line(combined_img, (int(p1[1]), int(p1[0])), (int(p2[1]), int(p2[0])), color, width, cv2.LINE_AA)
        
        # Add value labels periodically
        if i % 5 == 0:
            mid_x = int((p1[1] + p2[1]) / 2)
            mid_y = int((p1[0] + p2[0]) / 2)
            
            # Add text background
            text = f"{nsm:.1f}m"
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(combined_img, 
                         (mid_x - text_width//2 - 5, mid_y - text_height - 5),
                         (mid_x + text_width//2 + 5, mid_y + 5),
                         (0, 0, 0), -1)
            
            # Add text
            cv2.putText(combined_img, text, (mid_x - text_width//2, mid_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Add legend
    legend_y = 30
    cv2.putText(combined_img, "Red: Shoreline 1", (10, legend_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(combined_img, "Blue: Shoreline 2", (10, legend_y + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(combined_img, "Green: Accretion", (10, legend_y + 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(combined_img, "Red: Erosion", (10, legend_y + 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Add title
    title = f"Shoreline Change Analysis: {model_name}"
    cv2.putText(combined_img, title, (image1.shape[1]//2 - 200, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Add statistics
    stats_y = image1.shape[0] - 120
    avg_nsm = np.mean(nsm_values)
    max_accretion_val = np.max(nsm_values) if len(nsm_values) > 0 else 0
    max_erosion_val = abs(np.min(nsm_values)) if len(nsm_values) > 0 else 0
    
    cv2.putText(combined_img, f"Average change: {avg_nsm:.2f} m", 
                (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(combined_img, f"Max accretion: {max_accretion_val:.2f} m", 
                (10, stats_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(combined_img, f"Max erosion: {max_erosion_val:.2f} m", 
                (10, stats_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Save visualization
    output_path = os.path.join(model_dir, "shoreline_change_visualization.jpg")
    cv2.imwrite(output_path, combined_img)
    
    # Also save a version with black background for matplotlib
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "shoreline_change_visualization_matplotlib.png"), dpi=300, 
                facecolor='black', edgecolor='black')
    plt.close()
    
    # Calculate statistics for return
    avg_nsm = np.mean(nsm_values)
    max_accretion_val = np.max(nsm_values) if len(nsm_values) > 0 else 0
    max_erosion_val = abs(np.min(nsm_values)) if len(nsm_values) > 0 else 0
    
    stats = {
        "avg_nsm": avg_nsm,
        "max_nsm": max_accretion_val,
        "min_nsm": -max_erosion_val,
        "std_nsm": np.std(nsm_values),
        "accretion_percent": np.sum(nsm_values > 0) / len(nsm_values) * 100,
        "erosion_percent": np.sum(nsm_values < 0) / len(nsm_values) * 100
    }
    
    print(f"Average NSM: {stats['avg_nsm']:.2f} meters")
    print(f"Maximum accretion: {stats['max_nsm']:.2f} meters")
    print(f"Maximum erosion: {abs(stats['min_nsm']):.2f} meters")
    print(f"Standard Deviation: {stats['std_nsm']:.2f} meters")
    print(f"Accretion percentage: {stats['accretion_percent']:.1f}%")
    print(f"Erosion percentage: {stats['erosion_percent']:.1f}%")
    
    return stats

def extract_date_from_filename(filename):
    """
    Extract the date from a filename string that contains a date in format YYYY-MM-DD.
    
    Parameters:
    -----------
    filename : str
        The filename containing the date
        
    Returns:
    --------
    str
        The extracted date in format YYYY-MM-DD, or None if no date found
    """
    import re
    import os
    
    # Extract just the filename part if a full path is provided
    basename = os.path.basename(filename)
    
    # Use regex to find a date pattern in the filename
    date_pattern = r'(\d{4}-\d{2}-\d{2})'
    match = re.search(date_pattern, basename)
    
    if match:
        return match.group(1)
    else:
        return None

def calculate_time_interval_years(date1_str, date2_str):
    """
    Calculate the time interval between two dates in years.
    
    Parameters:
    -----------
    date1_str : str
        First date in format YYYY-MM-DD
    date2_str : str
        Second date in format YYYY-MM-DD
        
    Returns:
    --------
    float
        Time interval in years
    """
    from datetime import datetime
    
    if not date1_str or not date2_str:
        print("Warning: Missing date information, using default 1 year interval")
        return 1.0
    
    try:
        date1 = datetime.strptime(date1_str, '%Y-%m-%d')
        date2 = datetime.strptime(date2_str, '%Y-%m-%d')
        
        # Calculate difference in days and convert to years
        days_diff = abs((date2 - date1).days)
        years_diff = days_diff / 365.25
        
        return years_diff
    except Exception as e:
        print(f"Error calculating time interval: {str(e)}")
        print("Using default 1 year interval")
        return 1.0

# Modification to run_shoreline_analysis to extract dates from filenames
def run_shoreline_analysis(image1_path, image2_path, satelite1, satelite2, model_name):
    import cv2
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    import pandas as pd
    from datetime import datetime
    
    print(f"Reading segmented images from {model_name}")
    print(f"Image 1: {image1_path}")
    print(f"Image 2: {image2_path}")
    
    # Extract dates from filenames
    date1_str = extract_date_from_filename(image1_path)
    date2_str = extract_date_from_filename(image2_path)
    
    print(f"Extracted dates: {date1_str} and {date2_str}")
    
    # Calculate time interval in years
    time_interval_years = calculate_time_interval_years(date1_str, date2_str)
    print(f"Time interval: {time_interval_years:.2f} years")
    
    # Continue with the existing code...
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
    
    if image1 is None:
        raise ValueError(f"Could not read image at {image1_path}")
    if image2 is None:
        raise ValueError(f"Could not read image at {image2_path}")

    pixel_to_meter = 10.0
    model_dir = os.path.join("analysis_results", model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    plt.rcParams['figure.max_open_warning'] = 0
    
    try:
        print("Processing first mask")
        shoreline_points1 = np.argwhere(image1 > 0)
        shoreline1 = order_shoreline_points(shoreline_points1)

        print("Processing second mask")
        shoreline_points2 = np.argwhere(image2 > 0)
        shoreline2 = order_shoreline_points(shoreline_points2)

        print("Detecting sea direction using robust component analysis")
        sea_direction = identify_sea_component(satelite1, shoreline1)
        
        # Create binary masks for shorelines
        shoreline1_mask = np.zeros((image1.shape[0], image1.shape[1]), dtype=np.uint8)
        for point in shoreline1:
            y, x = int(point[0]), int(point[1])
            if 0 <= y < shoreline1_mask.shape[0] and 0 <= x < shoreline1_mask.shape[1]:
                shoreline1_mask[y, x] = 1
                
        shoreline2_mask = np.zeros((image2.shape[0], image2.shape[1]), dtype=np.uint8)
        for point in shoreline2:
            y, x = int(point[0]), int(point[1])
            if 0 <= y < shoreline2_mask.shape[0] and 0 <= x < shoreline2_mask.shape[1]:
                shoreline2_mask[y, x] = 1

        print(f"Generating transects with equal spacing along the shoreline")
        spacing_meters = 500
        
        # Generate truly perpendicular transects
        transect_results = generate_truly_perpendicular_transects(
            shoreline1,
            spacing_meters=spacing_meters,
            transect_length=min(image1.shape[0], image1.shape[1])/3,
            sea_direction=sea_direction,
            pixel_to_meter=pixel_to_meter,
            debug_output=True
        )
        
        # Extract just the transects from the results
        transects = transect_results["transects"]
        
        # Verify transects are perpendicular
        verification = verify_true_perpendicularity(shoreline1, transects)
        print(f"Verified transect perpendicularity:")
        print(f"  Average deviation from 90°: {verification['average_deviation']:.2f}°")
        print(f"  Maximum deviation from 90°: {verification['max_deviation']:.2f}°")
        print(f"  Perpendicular transects: {verification['percent_perpendicular']:.1f}%")

        print("Calculating NSM and EPR with perpendicular transects")
        # Use the extracted time interval for EPR calculation
        nsm_values, epr_values, intersection_points1, intersection_points2, valid_transects = calculate_shoreline_change_with_direction(
            shoreline1_mask, shoreline2_mask, transects,
            pixel_to_meter=pixel_to_meter,
            time_interval_years=time_interval_years
        )

        print("Generating change visualization")
        # Create combined image showing both shorelines
        combined_img = satelite1.copy()
        mask1 = np.zeros((satelite1.shape[0], satelite1.shape[1]), dtype=np.uint8)
        mask2 = np.zeros((satelite2.shape[0], satelite2.shape[1]), dtype=np.uint8)
        
        for point in shoreline1:
            y, x = int(point[0]), int(point[1])
            if 0 <= y < mask1.shape[0] and 0 <= x < mask1.shape[1]:
                mask1[y, x] = 255
        
        for point in shoreline2:
            y, x = int(point[0]), int(point[1])
            if 0 <= y < mask2.shape[0] and 0 <= x < mask2.shape[1]:
                mask2[y, x] = 255
        
        # Create RGB visualization
        combined_mask = np.zeros((satelite1.shape[0], satelite1.shape[1], 3), dtype=np.uint8)
        combined_mask[..., 0] = mask1  # Red channel for first shoreline
        combined_mask[..., 2] = mask2  # Blue channel for second shoreline
        
        # Blend with original image
        alpha = 0.7
        combined_img = cv2.addWeighted(combined_img, 1-alpha, combined_mask, alpha, 0)
        
        # Draw transects and change indicators
        max_accretion = max(max(nsm_values), 0) if len(nsm_values) > 0 else 0
        max_erosion = abs(min(min(nsm_values), 0)) if len(nsm_values) > 0 else 0
        
        for i, (transect, p1, p2, nsm) in enumerate(zip(valid_transects, intersection_points1, intersection_points2, nsm_values)):
            # Draw transect line
            start = (int(transect[0][1]), int(transect[0][0]))
            end = (int(transect[1][1]), int(transect[1][0]))
            cv2.line(combined_img, start, end, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Draw intersection points
            cv2.circle(combined_img, (int(p1[1]), int(p1[0])), 3, (0, 255, 255), -1)
            cv2.circle(combined_img, (int(p2[1]), int(p2[0])), 3, (255, 255, 0), -1)
            
            # Calculate line width based on magnitude of change
            abs_nsm = abs(nsm)
            max_val = max(max_accretion, max_erosion)
            width = 1 + 4 * (abs_nsm / max_val) if max_val > 0 else 1
            width = min(int(width), 10)  # Cap width
            
            # Draw change line with color based on direction
            if nsm > 0:  # Accretion
                color = (0, 255, 0)  # Green
            else:  # Erosion
                color = (0, 0, 255)  # Red
            
            cv2.line(combined_img, (int(p1[1]), int(p1[0])), (int(p2[1]), int(p2[0])), color, width, cv2.LINE_AA)
            
            # Add value labels periodically
            if i % 5 == 0:
                mid_x = int((p1[1] + p2[1]) / 2)
                mid_y = int((p1[0] + p2[0]) / 2)
                
                # Add text background
                text = f"{nsm:.1f}m"
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(combined_img, 
                             (mid_x - text_width//2 - 5, mid_y - text_height - 5),
                             (mid_x + text_width//2 + 5, mid_y + 5),
                             (0, 0, 0), -1)
                
                # Add text
                cv2.putText(combined_img, text, (mid_x - text_width//2, mid_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Add legend
        legend_y = 30
        cv2.putText(combined_img, "Red: Shoreline 1", (10, legend_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(combined_img, "Blue: Shoreline 2", (10, legend_y + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(combined_img, "Green: Accretion", (10, legend_y + 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(combined_img, "Red: Erosion", (10, legend_y + 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Add title with dates
        dates_info = f"{date1_str} to {date2_str}"
        title = f"Shoreline Change Analysis: {model_name} ({dates_info})"
        cv2.putText(combined_img, title, (satelite1.shape[1]//2 - 300, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Add statistics
        stats_y = satelite1.shape[0] - 120
        avg_nsm = np.mean(nsm_values)
        max_accretion_val = np.max(nsm_values) if len(nsm_values) > 0 else 0
        max_erosion_val = abs(np.min(nsm_values)) if len(nsm_values) > 0 else 0
        
        cv2.putText(combined_img, f"Average change: {avg_nsm:.2f} m", 
                    (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined_img, f"Max accretion: {max_accretion_val:.2f} m", 
                    (10, stats_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined_img, f"Max erosion: {max_erosion_val:.2f} m", 
                    (10, stats_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add time period information
        time_y = stats_y - 30
        cv2.putText(combined_img, f"Time period: {time_interval_years:.2f} years", 
                    (10, time_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save visualization
        output_path = os.path.join(model_dir, "shoreline_change_perpendicular_visualization.jpg")
        cv2.imwrite(output_path, combined_img)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, "shoreline_change_perpendicular_visualization_matplotlib.png"), dpi=300)
        plt.close()
        
        # Calculate statistics for return
        if len(nsm_values) > 0:
            stats = {
                "avg_nsm": np.mean(nsm_values),
                "max_nsm": np.max(nsm_values),
                "min_nsm": np.min(nsm_values),
                "std_nsm": np.std(nsm_values),
                "accretion_percent": np.sum(nsm_values > 0) / len(nsm_values) * 100,
                "erosion_percent": np.sum(nsm_values < 0) / len(nsm_values) * 100,
                "time_interval_years": time_interval_years,
                "date1": date1_str,
                "date2": date2_str
            }
        else:
            stats = {
                "avg_nsm": 0,
                "max_nsm": 0,
                "min_nsm": 0,
                "std_nsm": 0,
                "accretion_percent": 0,
                "erosion_percent": 0,
                "time_interval_years": time_interval_years,
                "date1": date1_str,
                "date2": date2_str
            }

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
                'EPR (m/year)': epr_values,
                'Perpendicular': True
            })
            results_df.to_csv(os.path.join(model_dir, "shoreline_perpendicular_results.csv"), index=False)
            
            # Add perpendicularity stats
            stats["avg_perpendicular_deviation"] = verification["average_deviation"]
            stats["max_perpendicular_deviation"] = verification["max_deviation"]
            stats["percent_perpendicular"] = verification["percent_perpendicular"]
            
            print(f"Average EPR: {stats['avg_epr']:.2f} meters/year")
            print(f"Time period: {time_interval_years:.2f} years ({date1_str} to {date2_str})")
            print(f"Results saved to {model_dir}/shoreline_perpendicular_results.csv")
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