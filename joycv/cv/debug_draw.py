import numpy as np
import copy
import cv2
def draw_rotated_bbox(img, rect, box, color,mask_index,grid_cord,result_obj):
    if rect is not None and box is not None:
        cv2.drawContours(img, [box], 0, color, 2)
        cv2.circle(img, box[0], 2, [0, 0, 255], 5) # Red 1
        cv2.circle(img, box[1], 2, [0, 255, 255], 5) # Yellow 2
        cv2.circle(img, box[2], 2, [255, 0, 0], 5) # Blue 3

        
        width = result_obj["width"]
        height = result_obj["height"]
        color = result_obj["color"]
        winkle_score = result_obj["winkle_score"]

        text=f"idx:{mask_index} grid({grid_cord[0]},{grid_cord[1]})"
        text2=f"w,h:{width:.2f},{height:.2f}"
        text3=f"color:{color:.2f},ws:{winkle_score:.2f}"
        #print(f"box[0]:{box[0]}")
        draw_text(img, text, box[0], (0, 255, 255), thickness=2);
        draw_text(img, text2, (box[0][0],box[0][1]+30), (0, 255, 255), thickness=2);
        draw_text(img, text3, (box[0][0],box[0][1]+60), (0, 255, 255), thickness=2);
    return img

def draw_text(img, text, position, color, thickness=2,font=cv2.FONT_HERSHEY_SIMPLEX):
    cv2.putText(img, text, position, font, 1, color, thickness)
    return img



def draw_dotted_line(img, pt1, pt2, color, thickness):
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
    pts = []
    for i in np.arange(0, dist, 10):
        r = i / dist
        x = int((1 - r) * pt1[0] + r * pt2[0])
        y = int((1 - r) * pt1[1] + r * pt2[1])
        pts.append((x, y))
    for p in pts:
        cv2.circle(img, p, thickness, color, -1)

def draw_debug_grid(img, grid):
    for i, grid_row in enumerate(grid):
        for j, cell in enumerate(grid_row):
            start_x, start_y = cell["grid"][0]
            end_x, end_y = cell["grid"][1]

            # Draw horizontal grid lines
            draw_dotted_line(img, (start_x, start_y), (end_x, start_y), (0, 0, 255 - j * 10), 1)
            draw_dotted_line(img, (start_x, end_y), (end_x, end_y), (0, 0, 255 - j * 10), 1)

            # Draw vertical grid lines
            draw_dotted_line(img, (start_x, start_y), (start_x, end_y), (0, 0, 255 - i * 10), 1)
            draw_dotted_line(img, (end_x, start_y), (end_x, end_y), (0, 0, 255 - i * 10), 1)

            # Draw circle at the center of the grid
            center_x, center_y = cell["center"]
            cv2.circle(img, (center_x, center_y), 5, (0, 0, 255), -1)

