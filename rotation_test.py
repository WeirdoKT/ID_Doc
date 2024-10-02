import cv2
import numpy as np
from ultralytics import YOLO
import math

def calculate_angle(box):
    # Assuming box coordinates are in the format [x1, y1, x2, y2, x3, y3, x4, y4]
    # where (x1, y1) is top-left, (x2, y2) is top-right, (x3, y3) is bottom-right, (x4, y4) is bottom-left
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    return angle

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def main():
    # Load the YOLOv8n model
    model = YOLO('yolov8n.pt')

    # Load the input image
    image_path = 'path/to/your/image.jpg'
    image = cv2.imread(image_path)

    # Run YOLOv8n prediction
    results = model(image)

    # Get the bounding box coordinates for the ID document class
    # Assuming the ID document class index is 0, adjust if necessary
    boxes = results[0].boxes.xyxy.cpu().numpy()
    if len(boxes) == 0:
        print("No ID document detected in the image.")
        return

    # Calculate the rotation angle based on the first detected box
    box = boxes[0]
    angle = calculate_angle(box)

    # Rotate the image
    rotated_image = rotate_image(image, -angle)  # Negative angle to correct the rotation

    # Save the rotated image
    output_path = 'path/to/output/rotated_image.jpg'
    cv2.imwrite(output_path, rotated_image)

    print(f"Image rotated by {angle:.2f} degrees and saved to {output_path}")

if __name__ == "__main__":
    main()
