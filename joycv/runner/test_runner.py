import numpy as np
import argparse
import math
import os
import cv2
import sys
from copy import deepcopy


sys.path.append("./joycv")
print(f"{sys.path}")


from mmdeploy_runtime import Detector
from datetime import datetime
from shapely.geometry import Polygon
from cv.grid_processing import get_grid,process_mask_grid,check_overlap,mask_center_in_grid
from cv.debug_draw import draw_debug_grid,draw_dotted_line,draw_text,draw_rotated_bbox
from cv.slice_image import slice_image,slice_single,save_single_slice
from cv.winkle_sobel import get_winkle_score,draw_winkle_debug,save_winkle

def parse_args():
    parser = argparse.ArgumentParser(
        description='show how to use sdk python api')
    parser.add_argument('input_path', help='path of an image or a folder containing images')
    parser.add_argument('--output_path', help='path of an image', default=None)
    parser.add_argument('--device_name', help='name of device, cuda or cpu', default='cuda')
    parser.add_argument('--model_path', help='path of mmdeploy SDK model dumped by model converter',
                        default='/opt/workspace/mmdeploy/work_dirs/rtmdet-tiny-ins-fullsize_single_cat_20230526')
    args = parser.parse_args()
    return args

import time

def process_image(detector, image_path, output_path, overlap_thresh=0.01, center_diff_thresh=0.35):
    print(f"reading {image_path}")
    img = cv2.imread(image_path)
    bboxes, labels, masks = detector(img)
    lowest_score = 1.0
    slice_image(img,bboxes, labels, masks,image_path,output_path)
    
    image_name = os.path.basename(image_path).split(".")[0]
    indices = [i for i in range(len(bboxes))]

    # grid = None
    # if grid is None:
    grid = get_grid(img, 4)       
    target_size=224
    confident_masks = []
    inconfident_masks = []
    mask_coordinates = []
    mask_scores = []
    sub_imgs = []
    for index, bbox, label_id in zip(indices, bboxes, labels):
        [left, top, right, bottom], score = bbox[0:4].astype(int), bbox[4]
        lowest_score = min(lowest_score, score)
        
        current_mask_object = masks[index]
        total_area = np.count_nonzero(current_mask_object)
        sub_img,offset_xy = slice_single(img,current_mask_object,bbox,target_size)
        


        if score < 0.4 or total_area< 7500:
            inconfident_masks.append(current_mask_object)
            continue
        else:
            mask_coordinates.append((left, top))
            mask_scores.append(score)
            sub_imgs.append((sub_img,offset_xy))
            confident_masks.append(current_mask_object)

    start = time.time()
    grid_with_masks, mask_data = process_mask_grid(confident_masks, grid, mask_coordinates,center_diff_thresh)
    time_for_process_mask_grid = time.time() - start

    start = time.time()
    overlapping_info = check_overlap(mask_data, overlap_thresh)
    time_for_check_overlap = time.time() - start

    detection_result = [''] * len(confident_masks)

    #print(f"{grid_with_masks}")
    start = time.time()
    for row_index, row in enumerate(grid_with_masks):
        for col_index, cell in enumerate(row):
            masks_in_grid = cell["masks_in_grid"] if "masks_in_grid" in cell else []
            masks_cdt_valid = cell["masks_cdt_valid"] if "masks_cdt_valid" in cell else []
            masks_center_in_grid = cell["masks_center_in_grid"] if "masks_center_in_grid" in cell else []
            
            if len(masks_cdt_valid) == 1:
                detection_result[masks_cdt_valid[0]] = 'valid'
            if len(masks_cdt_valid) > 1:
                for mask_id in masks_cdt_valid:
                    if detection_result[mask_id] != '':
                        print(f'Warning: Skipping mask {mask_id} for close which already has a result.{detection_result[mask_id]}')
                        continue
                    detection_result[mask_id] = 'close'
            for mask_id in masks_center_in_grid:
                if mask_id not in masks_cdt_valid:
                    if detection_result[mask_id] != '':
                        print(f'Warning: Skipping mask {mask_id} for distant which already has a result.{detection_result[mask_id]}')
                        continue
                    detection_result[mask_id] = 'distant'
            
            

    for i, overlap_info in enumerate(overlapping_info):
                if 'is_overlap' in overlap_info and overlap_info['is_overlap']:
                    detection_result[i] = 'overlapping'
   
    time_for_detection_result_filling = time.time() - start

    color_dict = {"valid": (0, 255, 0), "overlapping": (0, 0, 255), "close": (0, 0, 125), "distant": (255, 0, 0)}

    overlapping_img = img.copy()
    #print(f"detection_result:{detection_result}")
    for i, (mask_datum, status) in enumerate(zip(mask_data, detection_result)):
        if status != '':
            rect, box = mask_datum[1], mask_datum[2]

            #img = draw_rotated_bbox(img, rect, box, color_dict[status],i,mask_datum[4])

            if status == 'overlapping':
                # Draw the intersection polygon
                for overlap_info_index in range(len(overlapping_info)):
                    overlap_info=overlapping_info[overlap_info_index]
                    if 'is_overlap' in overlap_info and (not overlap_info['intersection'].is_empty) :
                        intersection_polygon = overlap_info['intersection']
                        intersection_points = np.array([point for point in intersection_polygon.exterior.coords], dtype=np.int32)
                        color=(0, 0, 255)
                        cv2.polylines(overlapping_img, [intersection_points], isClosed=True, color=color, thickness=2)
                        
                        text=f"idex:{overlap_info_index} v:{overlap_info['overlap_percentage']:.3f} "
                        #print(f"point:{intersection_points[0]}")
                        draw_point_idx=overlap_info_index%3
                        draw_color=(0, 0, 255-draw_point_idx*50)
                        draw_text(overlapping_img, text, (intersection_points[draw_point_idx][0]-40,intersection_points[draw_point_idx][1]), draw_color)

    
    #cv2.imwrite(os.path.join(output_path, f"{image_name}_overlapping.jpg"), overlapping_img)

    

    timing = {"process_mask_grid": time_for_process_mask_grid, "check_overlap": time_for_check_overlap, 
              "detection_result_filling": time_for_detection_result_filling}

    final_result = deepcopy(grid)

    # Loop through the detection result
    for idx, status in enumerate(detection_result):
        #if status == 'valid':
            # Get the mask center from mask_data
            mask_center = mask_data[idx][0]
            
            # Get the grid cell for this mask
            i, j = mask_center_in_grid(mask_center, grid)
            
            # If the mask is in the grid (i.e., i and j are not -1)
            if i != -1 and j != -1:
                # Get the mask information
                mask_info = mask_data[idx]
                landscape_Flag=mask_info[6] 
                bbox = mask_info[1]  # Rectangular bounding box
                width, height, color = mask_info[3]  # Width, height, and color of the mask
                sub_img,mask_offset  = sub_imgs[idx]
                winkle_score,ring_score=save_winkle(sub_imgs[idx],confident_masks[idx],landscape_Flag,output_path,image_name,True)
                winkle_score,ring_score=0,1
                save_single_slice(output_path,sub_img,image_name,mask_scores[idx],idx)
                # Update the corresponding cell in the final result
                final_result[i][j] = {
                    "valid_mask_idx": idx,
                    "status":status,
                    "width": width,
                    "height": height,
                    "color": color,
                    "winkle_score":winkle_score,
                    "area":total_area
                }
    

    
    #draw_debug_grid(img, grid)
    for i in range(len(final_result)):
        for j in range(len(final_result[i])):
            if "status" in final_result[i][j] and "valid_mask_idx" in final_result[i][j]:
                status=final_result[i][j]["status"]
                if status == 'valid':
                    
                    valid_mask_idx = final_result[i][j]["valid_mask_idx"]
                    rect, box, grid_cord= mask_data[valid_mask_idx][1], mask_data[valid_mask_idx][2],mask_data[valid_mask_idx][4]
                    img = draw_rotated_bbox(img, rect, box, color_dict[status], i, grid_cord, final_result[i][j])
                




    #cv2.imwrite(os.path.join(output_path, f"{image_name}.jpg"), img)
    return final_result, timing,lowest_score

        
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
    print(f"checking image path{args.input_path}")
    if os.path.isfile(args.input_path):
        print(f"image path is file")
        process_image(detector, args.input_path, args.output_path)
    elif os.path.isdir(args.input_path):
        total_times = []
        total_times_for_each_key = {"process_mask_grid": [], "check_overlap": [], "detection_result_filling": []}
        print(f"start to list files:{args.input_path}")
        for root, dirs, files in os.walk(args.input_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.bmp')):
                    input_image_path = os.path.join(root, file)
                    output_subdir = os.path.join(args.output_path, os.path.relpath(root, args.input_path))
                    
                    if not os.path.exists(output_subdir):
                        os.makedirs(output_subdir)
                    winkle_debug_path=os.path.join(output_subdir,"winkle_debug")
                    winkle_orderred_path=os.path.join(output_subdir,"winkle_orderred")
                    # if not os.path.exists(winkle_debug_path):
                    #     os.makedirs(winkle_debug_path)
                    # if not os.path.exists(winkle_orderred_path):
                    #     os.makedirs(winkle_orderred_path)
                    final_result, timing,lowest_score = process_image(detector, input_image_path, output_subdir)
                    
                    total_time = sum(timing.values())
                    total_times.append(total_time)

                    for key in timing:
                        total_times_for_each_key[key].append(timing[key])

                    #print(f"lowest score {lowest_score} in {file_name}")
                    if lowest_score <= 0.1:
                        file_name = file.split(".")[0]
                        low_confidence_image_path = os.path.join(low_confidence_path, f"{lowest_score:.2f}_{file_name}.png")
                        ori_img = cv2.imread(input_image_path)
                        cv2.imwrite(low_confidence_image_path, ori_img)
                        #copy input_image_path to low_confidence_image_path
                        print(f"low confidence {file_name}--> {low_confidence_image_path}")

        max_time = max(total_times)
        min_time = min(total_times)
        mean_time = sum(total_times) / len(total_times)

        print("Max time: ", max_time)
        print("Min time: ", min_time)
        print("Mean time: ", mean_time)

        # Now for each key in total_times_for_each_key calculate min, max, and mean
        for key, times in total_times_for_each_key.items():
            max_time_key = max(times)
            min_time_key = min(times)
            mean_time_key = sum(times) / len(times)

            print(f"Max time for {key}: {max_time_key:.4}")
            print(f"Min time for {key}: {min_time_key:.4}")
            print(f"Mean time for {key}: {mean_time_key:.4}" )

        
            #os.system(f"cp '{input_image_path}' '{low_confidence_image_path}'")
if __name__ == '__main__':
    
    main()