import numpy as np
import argparse
import math
import os
import cv2
from mmdeploy_runtime import Detector
from datetime import datetime
from shapely.geometry import Polygon



def parse_args():
    parser = argparse.ArgumentParser(
        description='show how to use sdk python api')
    parser.add_argument('input_path', help='path of an image or a folder containing images')
    parser.add_argument('--output_path', help='path of an image', default=None)
    parser.add_argument('--device_name', help='name of device, cuda or cpu', default='cuda')
    parser.add_argument('--model_path', help='path of mmdeploy SDK model dumped by model converter',
                        default='/opt/workspace/mmdeploy/work_dirs/rtmdet-tiny-ins-fullsize_single_cat_20230511')
    args = parser.parse_args()
    return args




def process_image(detector, image_path, output_path, overlap_thresh=0.01, center_diff_thresh=0.2):
    img = cv2.imread(image_path)
    bboxes, labels, masks = detector(img)
    lowest_score = 1.0

    indices = [i for i in range(len(bboxes))]
    mask_rects = []
    mask_centers = []

    for index, bbox, label_id in zip(indices, bboxes, labels):
        [left, top, right, bottom], score = bbox[0:4].astype(int), bbox[4]
        lowest_score = min(lowest_score, score)
        if score < 0.4:
            continue

        if masks[index].size:
            mask = masks[index]
            rect, box = get_rotated_bbox(mask, left, top)  # Pass the global coordinates to get_rotated_bbox

            mask_center = np.mean(box, axis=0)
            mask_centers.append(mask_center)
            mask_rects.append((rect, box))

    grid = draw_grid(img, 4)

    overlapping_masks,overlapping_values = check_overlap(mask_rects, overlap_thresh)
    centered_masks,centered_diff_ratios = check_center(mask_centers, grid, center_diff_thresh)

    for index, (rect, box) in enumerate(mask_rects):
        type = "ok"
        color = (0, 255, 0)  # Default green color
        if overlapping_masks[index]:
            type="overlap"
            color = (0, 0, 255)  # Red color for overlapping masks
        elif not centered_masks[index]:
            type="non-centered"
            color = (0, 255, 255)  # Yellow color for non-centered masks

        img = draw_rotated_bbox(img, rect, box, color)

        row, col = mask_in_grid(np.mean(box, axis=0),grid)
        #Draw the text
        angle = rect[-1]
        # if angle < -45:
        #     angle = 90 + angle
        oi = overlapping_values[index]["index"]
        ov = overlapping_values[index]["value"]
        try:
            cdr = centered_diff_ratios[index]
        except:
            cdr = -1
            print(f"index:{index} mask_len:{len(mask_rects)}  cdr_len:{len(centered_diff_ratios)} centered_diff_ratios:{centered_diff_ratios}")
        text = f"idx:{index} cdr:{cdr:.2f} oi:{oi:.2f} ov:{ov:.2f}"
        text_y = int(box[0][1])
        if(index%2==0):
            text_y = int(box[1][1])
        text_position = (int(box[0][0]), text_y)
        img = draw_text(img, text, text_position, color)

    image_name = os.path.basename(image_path)
    cv2.imwrite(os.path.join(output_path, image_name), img)
    return lowest_score





def mask_in_grid(mask_center, grid):
    for i, row in enumerate(grid):
        for j, col in enumerate(row):
            if col[0][0] <= mask_center[0] <= col[1][0] and col[0][1] <= mask_center[1] <= col[1][1]:
                return i, j
    return -1, -1







def main():
    args = parse_args()

    if args.output_path is None:
        output_base = os.path.basename(args.input_path)
        datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base_noext = os.path.dirname(args.input_path) 
        print(f"output_base_noext {output_base_noext}")
        args.output_path = f"{output_base_noext}_debug_draw_{datetime_str}"
    print(f"args.output_path {args.output_path}")
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    low_confidence_path = os.path.join(args.output_path, "low_confidence")
    if not os.path.exists(low_confidence_path):
        os.makedirs(low_confidence_path)

    detector = Detector(model_path=args.model_path, device_name=args.device_name, device_id=0)
    print(f"checking image path")
    if os.path.isfile(args.input_path):
        print(f"image path is file")
        process_image(detector, args.input_path, args.output_path)
    elif os.path.isdir(args.input_path):
        for root, dirs, files in os.walk(args.input_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.bmp')):
                    input_image_path = os.path.join(root, file)
                    #print(f"input_image_path {input_image_path}")
                    output_subdir = os.path.join(args.output_path, os.path.relpath(root, args.input_path))

                    if not os.path.exists(output_subdir):
                        os.makedirs(output_subdir)

                    lowest_score = process_image(detector, input_image_path, output_subdir)

                    if lowest_score < 0.4:
                        low_confidence_image_path = os.path.join(low_confidence_path, f"{lowest_score:.2f}_{file}")
                        #cv2.imwrite(low_confidence_image_path, img)
                        #copy input_image_path to low_confidence_image_path
                        #print(f"cp {input_image_path} {low_confidence_image_path}")
                        os.system(f"cp '{input_image_path}' '{low_confidence_image_path}'")

if __name__ == '__main__':
    main()
