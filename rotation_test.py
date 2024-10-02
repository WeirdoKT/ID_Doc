import cv2
import numpy as np
import math

def detect_and_correct_rotation(image_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Use Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = math.atan2(y2 - y1, x2 - x1) * 180.0 / np.pi
        angles.append(angle)
    
    # Calculate the median angle
    median_angle = np.median(angles)
    
    # If the median angle is close to 90 or -90, it's likely vertical text
    if abs(abs(median_angle) - 90) < 10:
        rotation_angle = 90 + median_angle
    else:
        rotation_angle = median_angle
    
    # Rotate the image
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated

# Usage
input_image_path = "path_to_your_image.jpg"
corrected_image = detect_and_correct_rotation(input_image_path)

# Save or display the result
cv2.imwrite("corrected_image.jpg", corrected_image)
cv2.imshow("Corrected Image", corrected_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
