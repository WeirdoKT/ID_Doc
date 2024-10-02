import cv2
import numpy as np
import pytesseract
from scipy.ndimage import rotate

def correct_id_rotation(image_path):
    # Read the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour (assuming it's the ID document)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get the minimum area rectangle
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    # Get the angle of rotation
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle
    
    # Rotate the image
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    # Check if the rotation is correct using OCR
    ocr_text = pytesseract.image_to_string(rotated)
    if len(ocr_text.strip()) < 10:  # If very little text is detected, try 90-degree rotations
        for angle in [90, 180, 270]:
            temp_rotated = rotate(rotated, angle)
            temp_ocr_text = pytesseract.image_to_string(temp_rotated)
            if len(temp_ocr_text.strip()) > len(ocr_text.strip()):
                rotated = temp_rotated
                ocr_text = temp_ocr_text
    
    return rotated

# Usage
input_image_path = "path_to_your_id_document.jpg"
corrected_image = correct_id_rotation(input_image_path)

# Save or display the result
cv2.imwrite("corrected_id_document.jpg", corrected_image)
cv2.imshow("Corrected ID Document", corrected_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
