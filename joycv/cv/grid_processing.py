import numpy as np
import copy
import cv2
from shapely.geometry import Polygon
def get_grid(img, row=4, offset_x=0, offset_y=0):
    height, width, _ = img.shape
    col = 8 if width > 1700 else 6
    grid_height = height // row
    grid_width = width // col
    middle_row = row / 2
    middle_col = col / 2
    row_offset_factor = -5
    col_offset_factor = -9

    grid = []
    for i in range(row):
        grid_row = []
        for j in range(col):
            row_offset = row_offset_factor * (i - middle_row)
            col_offset = col_offset_factor * (j - middle_col)

            start_x = int(j * grid_width + col_offset + offset_x)
            start_y = int(i * grid_height + row_offset + offset_y)

            # Calculate the start_x and start_y of the next cell
            next_col_offset = col_offset_factor * ((j + 1) - middle_col)
            next_start_x = int((j + 1) * grid_width + next_col_offset + offset_x)
            next_start_y = int((i + 1) * grid_height + row_offset + offset_y)

            # Use the start_x and start_y of the next cell as end_x and end_y of the current cell
            if j == col - 1:
                end_x = width + col_offset
            else:
                end_x = next_start_x
            if i == row - 1:
                end_y = height + row_offset
            else:
                end_y = next_start_y

            # Calculate the center of the grid
            center_x = int((start_x + end_x) // 2 + col_offset_factor * (j - middle_col + 0.5))
            center_y = int((start_y + end_y) // 2 + row_offset_factor * (i - middle_row + 0.5))

            grid_row.append({"grid": ((start_x, start_y), (end_x, end_y)), "center": (center_x, center_y)})

        grid.append(grid_row)

    return grid



def mask_in_grid(mask_center, grid):
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            if cell["grid"][0][0] <= mask_center[0] <= cell["grid"][1][0] and cell["grid"][0][1] <= mask_center[1] <= cell["grid"][1][1]:
                return i, j
    return -1, -1



def process_mask_grid(masks, grid,mask_coordinates, center_diff_thresh):
    grid_with_masks = copy.deepcopy(grid)
    mask_data = []
    for mask_index, mask in enumerate(masks):
        left,top = mask_coordinates[mask_index]
        rect, box = get_rotated_bbox(mask,left,top)
        if rect is not None:
            mask_center = (int(rect[0][0]), int(rect[0][1]))
            width, height = rect[1]
            if width < height:
                width, height = height, width
            row, col = mask_in_grid(mask_center, grid)

            # Calculate the average grayscale value from the mask area
            mask_pixels = mask[mask == 255]
            color = np.mean(mask_pixels)

            mask_data.append((mask_center, rect, box, (width, height, color), (row, col)))
        else:
            mask_data.append(None)

        if row >= 0 and col >= 0:
            grid_center = grid[row][col]["center"]

            distance_to_center = np.sqrt((mask_center[0] - grid_center[0]) ** 2 + (mask_center[1] - grid_center[1]) ** 2)
            grid_diag = np.sqrt((grid[row][col]["grid"][1][0] - grid[row][col]["grid"][0][0]) ** 2 + (grid[row][col]["grid"][1][1] - grid[row][col]["grid"][0][1]) ** 2)

            center_diff_ratio = distance_to_center / grid_diag
            #print(f"center_diff_ratio:{center_diff_ratio}")
            if "masks_center_in_grid" not in grid_with_masks[row][col]:
                grid_with_masks[row][col]["masks_center_in_grid"] = []

            if "masks_cdt_valid" not in grid_with_masks[row][col]:
                grid_with_masks[row][col]["masks_cdt_valid"] = []

            grid_with_masks[row][col]["masks_center_in_grid"].append(mask_index)

            if center_diff_ratio < center_diff_thresh:
                grid_with_masks[row][col]["masks_cdt_valid"].append(mask_index)

            if "masks_in_grid" not in grid_with_masks[row][col]:
                grid_with_masks[row][col]["masks_in_grid"] = []

            grid_with_masks[row][col]["masks_in_grid"].append(mask_index)

    return grid_with_masks, mask_data


def get_rotated_bbox(mask, x0, y0):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        cnt = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        # Add the global coordinate offsets
        box[:, 0] += x0
        box[:, 1] += y0
        # print(f"rect:{rect}")
        # print(f"box:{box}")
        return rect, box
    return None, None

def check_overlap(masks, mask_data, grid_with_masks, overlap_thresh):
    overlapping_info = [{}] * len(masks)

    for row in range(len(grid_with_masks)):
        for col in range(len(grid_with_masks[row])):
            #print(f"grid_with_masks[row][col]:{grid_with_masks[row][col]}")
            if "masks_in_grid" not in grid_with_masks[row][col]:
                continue
            masks_in_grid = grid_with_masks[row][col]["masks_in_grid"]

            for i, mask_index1 in enumerate(masks_in_grid):
                if overlapping_info[mask_index1].get('is_overlap', False):
                    continue
                #print(f"mask_data[mask_index1]: {mask_data[mask_index1]}")
                box1 = mask_data[mask_index1][2]
                #print(f"box1: {box1}")

                for j, mask_index2 in enumerate(masks_in_grid):
                    if i == j or overlapping_info[mask_index2].get('is_overlap', False):
                        continue

                    box2 = mask_data[mask_index2][2]
                    intersection, is_overlap, overlap_percentage = is_bbox_overlap(box1, box2, (mask_index1, mask_index2), (col, row), overlap_thresh)

                    if is_overlap:
                        overlapping_info[mask_index1] = {
                            'intersection': intersection,
                            'is_overlap': is_overlap,
                            'overlap_percentage': overlap_percentage
                        }
                        overlapping_info[mask_index2] = {
                            'intersection': intersection,
                            'is_overlap': is_overlap,
                            'overlap_percentage': overlap_percentage
                        }

    return overlapping_info



def is_bbox_overlap(bbox1, bbox2, mask_indices, grid_coords, overlap_thresh):
    is_overlapping = False
    box1_points = bbox1
    box2_points = bbox2
    #print(f"box1_points: {box1_points}, box2_points: {box2_points}")
    # Create Polygon objects from the points of the rectangles
    polygon1 = Polygon(box1_points)
    polygon2 = Polygon(box2_points)
    
    # Calculate the intersection polygon
    intersection = polygon1.intersection(polygon2)

    # Check if the intersection area is empty
    if intersection.is_empty:
        return intersection, is_overlapping, 0

    # Calculate the areas of the rectangles and their intersection
    area1 = polygon1.area
    area2 = polygon2.area
    intersection_area = intersection.area

    # Calculate the union area
    union_area = area1 + area2 - intersection_area

    # Calculate the overlap percentage
    overlap_percentage = intersection_area / union_area
    is_overlapping = overlap_percentage > overlap_thresh

    # if is_overlapping:
    #     print(f"intersection_area: {intersection_area}, union_area: {union_area}, is_overlapping: {is_overlapping}, overlap_percentage: {overlap_percentage:.4f}, mask_indices: {mask_indices}, grid_coords: {grid_coords}")
    
    return intersection, is_overlapping, overlap_percentage

import cv2
import numpy as np

def intersection_area(box1_points, box2_points):
    box1_points = box1_points.astype(np.int32)
    box2_points = box2_points.astype(np.int32)
    _, intersection_points = cv2.intersectConvexConvex(box1_points, box2_points)

    if len(intersection_points) > 0:
        intersection_area = cv2.contourArea(intersection_points)
    else:
        intersection_area = 0

    return intersection_area, intersection_points

def is_bbox_overlap_cv2(bbox1, bbox2, mask_indices, grid_coords, overlap_thresh):
    is_overlapping = False
    box1_points = np.array(bbox1[1])
    box2_points = np.array(bbox2[1])

    intersection_area_value, intersection_points = intersection_area(box1_points, box2_points)

    area1 = cv2.contourArea(box1_points)
    area2 = cv2.contourArea(box2_points)

    union_area = area1 + area2 - intersection_area_value

    if union_area == 0:
        return intersection_points, is_overlapping, 0

    overlap_percentage = intersection_area_value / union_area
    is_overlapping = overlap_percentage > overlap_thresh

    if is_overlapping:
        print(f"intersection_area: {intersection_area_value}, union_area: {union_area}, is_overlapping: {is_overlapping}, overlap_percentage: {overlap_percentage:.4f}, mask_indices: {mask_indices}, grid_coords: {grid_coords}")

    return intersection_points, is_overlapping, overlap_percentage
