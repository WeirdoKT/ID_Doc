import fitz
import numpy as np
import cv2
from PIL import Image
import pytesseract
import pyperclip
from scipy.ndimage import rotate
import matplotlib.pyplot as plt

def extract_pdf_page(pdf_path, page_number):
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_number - 1)
    pix = page.get_pixmap(dpi=300)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return np.array(img)

def adaptive_threshold(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

def improved_deskew(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    
    if lines is None:
        return image

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180. / np.pi
        angles.append(angle)
    
    median_angle = np.median(angles)
    return rotate(image, median_angle, reshape=False, order=3, mode='constant', cval=255)

def remove_grid(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Remove horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
    remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(image, [c], -1, (255,255,255), 5)

    # Remove vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,40))
    remove_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(image, [c], -1, (255,255,255), 5)
    
    return image

def enhance_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply bilateral filter to smooth image and preserve edges
    smoothed = cv2.bilateralFilter(gray, 9, 75, 75)

    # Sharpen the image using a kernel
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(smoothed, -1, kernel)

    # Increase contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(sharpened)

    return enhanced

def process_image(image):
    # Enhance the image
    enhanced = enhance_image(image)
    
    # Adaptive thresholding
    binary = adaptive_threshold(enhanced)
    
    # Deskew
    deskewed = improved_deskew(binary)
    
    # Remove grid
    cleaned = remove_grid(deskewed)
    
    return cleaned

def extract_text_from_image(image, config="--oem 1 --psm 6", language="eng"):
    # Use Tesseract to extract text
    extracted_text = pytesseract.image_to_string(image, config=config, lang=language)
    return extracted_text

def post_process_text(text):
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # You can add more post-processing steps here, such as:
    # - Correcting common OCR errors
    # - Removing specific characters or patterns
    # - Applying any domain-specific cleaning
    
    return text

def main():
    try:
        # Get input parameters from clipboard
        file_path, language, config, pdf, page_num = pyperclip.paste().split("|")
        page_num = int(page_num)
        print(f"File: {file_path}\nLanguage: {language}\nConfig: {config}\nPDF/IMG: {pdf}")

        # Load the image
        if pdf == "PDF":
            image = extract_pdf_page(file_path, page_num)
        else:
            image = cv2.imread(file_path)

        # Process the image
        processed_image = process_image(image)

        # Extract text
        output_text = extract_text_from_image(processed_image, language=language, config=config)

        # Post-process the extracted text
        cleaned_text = post_process_text(output_text)

        print(cleaned_text)
        pyperclip.copy(cleaned_text)

        # Optionally, save intermediate results for debugging
        # cv2.imwrite("processed_image.png", processed_image)

    except Exception as e:
        error_message = f"Python error - {str(e)}"
        print(error_message)
        pyperclip.copy(error_message)

if __name__ == "__main__":
    main()
