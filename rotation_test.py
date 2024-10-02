import cv2
import numpy as np
import pytesseract
from pytesseract import Output

def detect_and_correct_rotation(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to preprocess the image
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # Detect orientation and rotate if necessary
    osd = pytesseract.image_to_osd(thresh, output_type=Output.DICT)
    angle = float(osd["rotate"])

    if angle != 0:
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    else:
        rotated = image

    return rotated

# Usage
input_image_path = "path_to_your_image.jpg"
corrected_image = detect_and_correct_rotation(input_image_path)

# Save or display the result
cv2.imwrite("corrected_image.jpg", corrected_image)
cv2.imshow("Corrected Image", corrected_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
