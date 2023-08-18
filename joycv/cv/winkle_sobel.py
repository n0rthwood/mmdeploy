import cv2
import os
import numpy as np
import argparse
import timeit
import shutil
import sys
if __name__ == "__main__":
    sys.path.append("../joycv")
    print(f"{sys.path}")



def get_winkle_score(img_with_offset, mask,landscape_flag=False):
    img,mask_offset = img_with_offset

    x_offset, y_offset = mask_offset

    # Create an empty mask of the same size as the image
    full_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    
    # Place the mask at the correct position on the full mask
    y_length = min(mask.shape[0], full_mask.shape[0] - y_offset)
    x_length = min(mask.shape[1], full_mask.shape[1] - x_offset)

    # Resize the mask if needed
    if mask.shape[0] != y_length or mask.shape[1] != x_length:
        mask = cv2.resize(mask, (x_length, y_length))

    full_mask[y_offset:y_offset+y_length, x_offset:x_offset+x_length] = mask
    mask = full_mask
    #print(img.shape,mask.shape)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    #mask = cv2.bitwise_not(mask)
    #mask = cv2.erode(mask, kernel_open, iterations = 1)
    #total_pixels = np.count_nonzero(mask)

    if(landscape_flag):
        sobel_ops = cv2.Sobel(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F, 0, 1, ksize=5)
    else:
        sobel_ops = cv2.Sobel(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F, 1, 0, ksize=5)
    sobel_ops = cv2.normalize(sobel_ops, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # Assuming 'img' is your image and 'mask' is your mask

    masked_sobel_ops = cv2.bitwise_and(sobel_ops, sobel_ops, mask=mask)
    masked_sobel_ops_blackhat = cv2.morphologyEx(masked_sobel_ops, cv2.MORPH_BLACKHAT, kernel_open)
    #mask = cv2.erode(mask, kernel_open, iterations = 1)

        # Erode the mask first
    kernel = np.ones((8,8), np.uint8)  # You can adjust the size of the kernel as needed
    eroded_mask = cv2.erode(mask, kernel, iterations=1)

    kernel = np.ones((8,8), np.uint8)  # You can adjust the size of the kernel as needed
    double_eroded_mask = cv2.erode(eroded_mask, kernel, iterations=4)

        # Create an inverse of the double_eroded_mask
    inverse_double_eroded_mask = cv2.bitwise_not(double_eroded_mask)

    # Perform a bitwise AND operation between the eroded_mask and the inverse_double_eroded_mask
    ring_mask = cv2.bitwise_and(eroded_mask, inverse_double_eroded_mask)


    # Apply the eroded mask to the image
    eroded_mask_sobel_ops_blackhat = cv2.bitwise_and(masked_sobel_ops_blackhat, masked_sobel_ops_blackhat, mask=eroded_mask)
    double_eroded_mask_sobel_ops_blackhat = cv2.bitwise_and(masked_sobel_ops_blackhat, masked_sobel_ops_blackhat, mask=double_eroded_mask)
    ring_mask_sobel_ops_blackhat = cv2.bitwise_and(masked_sobel_ops_blackhat, masked_sobel_ops_blackhat, mask=ring_mask)

    # weights = np.zeros_like(ring_mask_sobel_ops_blackhat)

    # weights[np.where((0 <= ring_mask_sobel_ops_blackhat) & (ring_mask_sobel_ops_blackhat < 10))] = 0
    # weights[np.where((10 <= ring_mask_sobel_ops_blackhat) & (ring_mask_sobel_ops_blackhat < 30))] = 20
    # weights[np.where((30 <= ring_mask_sobel_ops_blackhat) & (ring_mask_sobel_ops_blackhat < 50))] = 50
    # weights[np.where((50 <= ring_mask_sobel_ops_blackhat) & (ring_mask_sobel_ops_blackhat < 80))] = 100
    # weights[np.where((80 <= ring_mask_sobel_ops_blackhat) & (ring_mask_sobel_ops_blackhat < 100))] = 150
    # weights[np.where((100 <= ring_mask_sobel_ops_blackhat) & (ring_mask_sobel_ops_blackhat < 130))] = 230
    # weights[np.where((130 <= ring_mask_sobel_ops_blackhat) & (ring_mask_sobel_ops_blackhat < 160))] = 240
    # weights[np.where((160 <= ring_mask_sobel_ops_blackhat) & (ring_mask_sobel_ops_blackhat <= 200))] = 255
    # ring_mask_sobel_ops_blackhat = weights
    brightness1 = np.sum(eroded_mask_sobel_ops_blackhat)
    #brightness2 = np.sum(ring_mask_sobel_ops_blackhat)
    brightness2 = calculate_brightness(ring_mask_sobel_ops_blackhat,30)
    # Calculate the brightness of the masked area
    #brightness = np.sum(masked_sobel_ops_blackhat)

    # Get the total possible brightness if all pixels were white
    total_possible_brightness1 = np.sum(double_eroded_mask) * 255
    total_possible_brightness2 = np.sum(ring_mask) * 255
    
    # Get the percentage of the actual brightness to the total possible brightness
    winkle_score = ((brightness1/total_possible_brightness1) ) * 100*100
    ring_score = (brightness2/total_possible_brightness2)*10000
    #masked_sobel_ops = cv2.bitwise_and(masked_sobel_ops_blackhat, masked_sobel_ops_blackhat, mask=mask)

   
    # #dark_pixels = np.count_nonzero(binary_img)

    # kernel = np.ones((5,5),np.uint8)
    # dilated_img = cv2.dilate(binary_img, kernel, iterations = 1)
    # eroded_img = cv2.erode(dilated_img, kernel, iterations = 1)

    # contours, _ = cv2.findContours(eroded_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #percentage = dark_pixels/total_pixels*100

    return (winkle_score,ring_score,brightness1,brightness2,total_possible_brightness1,total_possible_brightness2),masked_sobel_ops_blackhat,eroded_mask_sobel_ops_blackhat, double_eroded_mask_sobel_ops_blackhat, ring_mask_sobel_ops_blackhat

def calculate_brightness(mask, pixels):
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assuming that the largest contour corresponds to the object of interest
    # sort contours by area and take the largest one
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Get the rotated bounding box of the largest contour
    rotated_bbox = cv2.minAreaRect(contours[0])

    # The `cv2.minAreaRect` function returns values in the format ((center_x, center_y), (width, height), angle)
    center, (width, height), angle = rotated_bbox

    # If width is less than height, swap them and adjust the angle
    if width < height:
        width, height = height, width
        angle += 90  # adjust angle

    # Create masks for the rectangles
    mask_a = np.zeros_like(mask)
    mask_b = np.zeros_like(mask)

    # Calculate the coordinates of the rectangles
    delta_x = np.cos(np.radians(angle)) * width/2
    delta_y = np.sin(np.radians(angle)) * width/2

    center_a = (center[0] - delta_x, center[1] - delta_y)
    center_b = (center[0] + delta_x, center[1] + delta_y)

    rect_a = (center_a, (pixels, height), angle)
    rect_b = (center_b, (pixels, height), angle)

    # Convert rectangles to box format for drawing
    box_a = cv2.boxPoints(rect_a).astype(np.int0)
    box_b = cv2.boxPoints(rect_b).astype(np.int0)

    # Draw rectangles on the masks
    cv2.drawContours(mask_a, [box_a], 0, (255), thickness=-1)
    cv2.drawContours(mask_b, [box_b], 0, (255), thickness=-1)

    # Calculate brightness within these areas
    brightness_left = np.sum(mask * (mask_a / 255))
    brightness_right = np.sum(mask * (mask_b / 255))

    # Draw the rectangles on the mask
    cv2.drawContours(mask, [box_a, box_b], -1, (255), 1)

    return brightness_left + brightness_right




def draw_winkle_debug(img, stats,masked_sobel_ops_blackhat,eroded_mask_sobel_ops_blackhat, double_eroded_mask_sobel_ops_blackhat, ring_mask_sobel_ops_blackhat):
    (winkle_score,ring_score,brightness1,brightness2,total_possible_brightness1,total_possible_brightness2)=stats
    double_eroded_mask_sobel_ops_blackhat_BGR = cv2.cvtColor(double_eroded_mask_sobel_ops_blackhat, cv2.COLOR_GRAY2BGR)
    ring_mask_sobel_ops_blackhat_BGR = cv2.cvtColor(ring_mask_sobel_ops_blackhat, cv2.COLOR_GRAY2BGR)
    masked_sobel_ops_blackhat = cv2.cvtColor(masked_sobel_ops_blackhat, cv2.COLOR_GRAY2BGR)

    #cv2.drawContours(img_bgr, contours, -1, (0,255,0), 2)

    text_canvas = np.zeros_like(img)
    cv2.putText(text_canvas, f"total brt1: {total_possible_brightness1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(text_canvas, f"total brt2: {total_possible_brightness2}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(text_canvas, f"brightness1: {brightness1}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(text_canvas, f"brightness2: {brightness2},{brightness2*2}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(text_canvas, f"score1: {brightness1/total_possible_brightness1*10000}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(text_canvas, f"score1: {brightness2/total_possible_brightness2*10000}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(text_canvas, f"winkle_score: {winkle_score:.4f}%", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    final_image = np.hstack((img, masked_sobel_ops_blackhat,ring_mask_sobel_ops_blackhat_BGR,double_eroded_mask_sobel_ops_blackhat_BGR , text_canvas))
    
    return final_image,img

def process_image(image_path, output_dir, dark_pixel_threshold):
    img = cv2.imread(image_path)  # Read the image in color
    base = os.path.basename(image_path)
    filename = os.path.splitext(base)[0]

    mask = cv2.inRange(cv2.cvtColor(img,cv2.COLOR_BGR2RGB), np.array([52, 119, 113]), np.array([255, 255, 255]))

    # Apply morphological operations to clean up mask
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    percentage, binary_img, masked_sobel_ops, masked_sobel_ops_blackhat, contours = get_winkle_score(img, mask)

    final_image = draw_winkle_debug(img, mask, percentage, binary_img, masked_sobel_ops, masked_sobel_ops_blackhat, contours)
    
    cv2.imwrite(os.path.join(output_dir, f"{percentage:.2f}_"+filename + "_combined.png"), final_image)
    return percentage


def process_images(input_dir, output_dir, dark_pixel_threshold):
    if  os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    percentages = []
    timing = []
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
            print(f"prcessing file :{filename}")
            start_time = timeit.default_timer()
            p = process_image(os.path.join(input_dir, filename), output_dir, dark_pixel_threshold)
            elapsed = timeit.default_timer() - start_time
            percentages.append(p)
            timing.append(elapsed)

    # calculate min, max, and mean for percentages
    min_val_percentages = np.min(percentages)
    max_val_percentages = np.max(percentages)
    mean_val_percentages = np.mean(percentages)

    # print them
    print("Percentages:")
    print("Minimum value: ", min_val_percentages)
    print("Maximum value: ", max_val_percentages)
    print("Mean value: ", mean_val_percentages)

    # calculate min, max, and mean for timing
    min_val_timing = np.min(timing)
    max_val_timing = np.max(timing)
    mean_val_timing = np.mean(timing)

    # print them
    print("\nTiming:")
    print("Minimum value: ", min_val_timing)
    print("Maximum value: ", max_val_timing)
    print("Mean value: ", mean_val_timing)

    print("Total count:" ,len(timing))


def save_winkle(img_with_offset, mask,landscape_flag,output_path,image_name,save_winkle_flag=False):
        winkle_stats,masked_sobely_blackhat,eroded_mask_sobely_blackhat, double_eroded_mask_sobely_blackhat, ring_mask_sobely_blackhat=get_winkle_score(img_with_offset, mask,landscape_flag)
        (winkle_score,ring_score,brightness1,brightness2,total_possible_brightness1,total_possible_brightness2)=winkle_stats
        debug_winkle_score_debug,ori_winkle_img = draw_winkle_debug(img_with_offset[0], winkle_stats,masked_sobely_blackhat,eroded_mask_sobely_blackhat, double_eroded_mask_sobely_blackhat, ring_mask_sobely_blackhat)
        total_area = np.count_nonzero(mask)
        
        

        sub_winkle_path="not_dry"
        if(ring_score>0.4 and winkle_score>3.9):
            sub_winkle_path="dry"
        total_winkle=winkle_score+ring_score
        
        if save_winkle_flag:
            winkle_debug_path=os.path.join(output_path,"winkle_debug")
            winkle_order_path=os.path.join(output_path,"winkle_orderred")
            if(not os.path.exists(os.path.join(winkle_debug_path,sub_winkle_path))):
                os.makedirs(os.path.join(winkle_debug_path,sub_winkle_path),exist_ok=True)
                print(f"creating {os.path.join(winkle_debug_path,sub_winkle_path)}")
            if(not os.path.exists(os.path.join(winkle_order_path,sub_winkle_path))):
                os.makedirs(os.path.join(winkle_order_path,sub_winkle_path),exist_ok=True)
                print(f"creating {os.path.join(winkle_order_path,sub_winkle_path)}")
            
            cv2.imwrite(os.path.join(winkle_debug_path,sub_winkle_path, f"{total_winkle:.4f}_{ring_score:.4f}_{winkle_score:.4f}_"+f"{image_name}" + "_combined.jpg"), debug_winkle_score_debug)
            cv2.imwrite(os.path.join(winkle_order_path,sub_winkle_path, f"{total_winkle:.4f}_{ring_score:.4f}_{winkle_score:.4f}_"+f"{image_name}" + ".jpg"), ori_winkle_img)
            
        return winkle_score,ring_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="/nas/win_essd/UAE_sliced_256/pd_train_candidate/mejood_train/train/dry/medjool_2nd_dry/")
    parser.add_argument("--output_dir", default="/opt/images/dry_edge_debug1/")
    parser.add_argument("--dark_pixel_threshold", type=int, default=150)
    args = parser.parse_args()
    
    
    process_images(args.input_dir, args.output_dir, args.dark_pixel_threshold)
