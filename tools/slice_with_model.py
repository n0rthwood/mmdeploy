import argparse
import os
import glob
import math
import time
from datetime import datetime

import cv2
import numpy as np
from mmdeploy_runtime import Detector


def parse_args():
    parser = argparse.ArgumentParser(
        description='show how to use sdk python api')
    parser.add_argument('device_name', help='name of device, cuda or cpu')
    parser.add_argument(
        'model_path',
        help='path of mmdeploy SDK model dumped by model converter')
    parser.add_argument('image_path', help='path of the image folder')
    parser.add_argument('--output_path', help='path of the output folder', default=None)
    args = parser.parse_args()
    return args


def process_image(detector, img_path, output_path):
    img = cv2.imread(img_path)
    start_time = time.time()
    bboxes, labels, masks = detector(img)
    inference_time = time.time() - start_time
    target_size=256
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    indices = [i for i in range(len(bboxes))]
    for index, bbox, label_id in zip(indices, bboxes, labels):
        [left, top, right, bottom], score = bbox[0:4].astype(int), bbox[4]
        if score < 0.3:
            continue

        if masks[index].size:
            mask = masks[index]
            x0 = int(max(math.floor(bbox[0]) - 1, 0))
            y0 = int(max(math.floor(bbox[1]) - 1, 0))
            mask_img = img[y0:y0 + mask.shape[0], x0:x0 + mask.shape[1]]

            masked_image = cv2.bitwise_and(mask_img, mask_img, mask=mask)

            # Replace black area with the specified background color
            background_color = np.full(mask_img.shape, [113, 119, 52], dtype=np.uint8)
            background_mask = cv2.bitwise_and(background_color, background_color, mask=cv2.bitwise_not(mask))
            masked_image = cv2.add(masked_image, background_mask)

            # Scale down if necessary
            if max(masked_image.shape) > target_size:
                print(f"Scale down image {img_name}_{index} to target_size")
                scale_factor = target_size / max(masked_image.shape)
                new_size = (int(masked_image.shape[1] * scale_factor), int(masked_image.shape[0] * scale_factor))
                masked_image = cv2.resize(masked_image, new_size)

            # Place the masked image on a 224x224 canvas with the background color
            canvas = np.full((target_size, target_size, 3), [113, 119, 52], dtype=np.uint8)
            x_offset = (target_size - masked_image.shape[1]) // 2
            y_offset = (target_size - masked_image.shape[0]) // 2
            canvas[y_offset:y_offset + masked_image.shape[0], x_offset:x_offset + masked_image.shape[1]] = masked_image

            cv2.imwrite(os.path.join(output_path, f"{img_name}_{index}.png"), canvas)



    return inference_time


def main():
    args = parse_args()

    if not args.output_path:
        output_folder_name = f"{os.path.basename(args.image_path)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        args.output_path = os.path.join(args.image_path, output_folder_name)
    os.makedirs(args.output_path, exist_ok=True)

    detector = Detector(
        model_path=args.model_path, device_name=args.device_name, device_id=0)

    image_files = glob.glob(os.path.join(args.image_path, "*"))
    inference_times = []

    for img_path in image_files:
        if os.path.isfile(img_path):
            inference_time = process_image(detector, img_path, args.output_path)
            inference_times.append(inference_time)

    max_time = max(inference_times)
    min_time = min(inference_times)
    mean_time = sum(inference_times) / len(inference_times)

    print(f"Max Inference Time: {max_time:.2f}s")
    print(f"Min Inference Time: {min_time:.2f}s")
    print(f"Mean Inference Time: {mean_time:.2f}s")
    print(f"Output path: {args.output_path}")

if __name__ == '__main__':
    main()
