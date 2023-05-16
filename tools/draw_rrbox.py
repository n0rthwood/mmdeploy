import numpy as np
import argparse
import math
import os
import cv2
from mmdeploy_runtime import Detector


def parse_args():
    parser = argparse.ArgumentParser(
        description='show how to use sdk python api')
    parser.add_argument('device_name', help='name of device, cuda or cpu')
    parser.add_argument(
        'model_path',
        help='path of mmdeploy SDK model dumped by model converter')
    parser.add_argument('image_path', help='path of an image')
    parser.add_argument('output_path', help='path of an image')
    args = parser.parse_args()
    return args
def main():
    args = parse_args()

    img = cv2.imread(args.image_path)
    detector = Detector(
        model_path=args.model_path, device_name=args.device_name, device_id=0)
    bboxes, labels, masks = detector(img)

    indices = [i for i in range(len(bboxes))]
    for index, bbox, label_id in zip(indices, bboxes, labels):
        [left, top, right, bottom], score = bbox[0:4].astype(int), bbox[4]
        if score < 0.4:
            continue

        if masks[index].size:
            mask = masks[index]
            blue, green, red = cv2.split(img)

            x0 = int(max(math.floor(bbox[0]) - 1, 0))
            y0 = int(max(math.floor(bbox[1]) - 1, 0))
            mask_img = blue[y0:y0 + mask.shape[0], x0:x0 + mask.shape[1]]
            cv2.bitwise_or(mask, mask_img, mask_img)
            img = cv2.merge([blue, green, red])
            total_pixels = np.count_nonzero(mask)

            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Calculate the rotated bounding box
            if len(contours) > 0:
                cnt = max(contours, key=cv2.contourArea)
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.intp(box)


                # Add the global coordinate offsets
                box[:, 0] += x0
                box[:, 1] += y0

                # Draw the rotated bounding box
                cv2.drawContours(img, [box], 0, (0, 255, 0), 2)

                # Display the rotation angle
                angle = rect[-1]
                if angle < -45:
                    angle = 90 + angle
                text_pos = (int(rect[0][0]) + x0-30, int(rect[0][1]) + y0)
                cv2.putText(img, f"area:{total_pixels} s:{score:.2f} angle:{angle:.2f}", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                #cv2.putText(img, f"area:{total_pixels} s:{score:.2f} angle:{angle:.2f}", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    image_name= os.path.basename(args.image_path)
    cv2.imwrite(os.path.join(args.output_path, image_name), img)

if __name__ == '__main__':
    main()
