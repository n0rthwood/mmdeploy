import os
import glob
import math
import time
from datetime import datetime

import cv2
import numpy as np

def slice_single(img,mask,bbox,target_size):
    x0 = int(max(math.floor(bbox[0]) - 1, 0))
    y0 = int(max(math.floor(bbox[1]) - 1, 0))
    mask_img = img[y0:y0 + mask.shape[0], x0:x0 + mask.shape[1]]

    masked_image = cv2.bitwise_and(mask_img, mask_img, mask=mask)
    #masked_image=mask_img
    # Replace black area with the specified background color
    background_color = np.full(mask_img.shape, [113, 119, 52], dtype=np.uint8)
    background_mask = cv2.bitwise_and(background_color, background_color, mask=cv2.bitwise_not(mask))
    masked_image = cv2.add(masked_image, background_mask)

    # Scale down if necessary
    if max(masked_image.shape) > target_size:
        #print(f"Scale down image {img_name}_{index} to target_size")
        scale_factor = target_size / max(masked_image.shape)
        new_size = (int(masked_image.shape[1] * scale_factor), int(masked_image.shape[0] * scale_factor))
        masked_image = cv2.resize(masked_image, new_size)

    # Place the masked image on a 224x224 canvas with the background color
    canvas = np.full((target_size, target_size, 3), [113, 119, 52], dtype=np.uint8)
    x_offset = (target_size - masked_image.shape[1]) // 2
    y_offset = (target_size - masked_image.shape[0]) // 2
    canvas[y_offset:y_offset + masked_image.shape[0], x_offset:x_offset + masked_image.shape[1]] = masked_image
    return canvas,(x_offset,y_offset)

def slice_image(img,bboxes, labels, masks, img_path, output_path):
    
    start_time = time.time()
    
    inference_time = time.time() - start_time
    target_size=224
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    indices = [i for i in range(len(bboxes))]
    
    for index, bbox, label_id in zip(indices, bboxes, labels):
        [left, top, right, bottom], score = bbox[0:4].astype(int), bbox[4]
        if score < 0.4:
            continue

        if masks[index].size:
            mask = masks[index]
            sub_img,offset_xy = slice_single(img,mask,bbox,target_size)
            save_path=output_path#os.path.join(output_path,"slice")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_path = os.path.join(save_path, f"{img_name}{score}_{index}.png")
            print(f"slicing {save_path}")
            cv2.imwrite(save_path, sub_img)



    return inference_time

global_counter=0
import math

def save_single_slice(output_path,sub_img,img_name,score,index):
    global global_counter
    
    sub_path=math.floor(global_counter/3000)
    save_path=os.path.join(output_path,f"group_{sub_path}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = os.path.join(save_path, f"{img_name}{score}_{index}.png")
    print(f"saving sliced: {save_path}")
    cv2.imwrite(save_path, sub_img)
    global_counter=global_counter+1